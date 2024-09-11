#ifndef KMEANS_INITIALIZE_VARIANCE_PARTITION_HPP
#define KMEANS_INITIALIZE_VARIANCE_PARTITION_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <random>
#include <cstdint>

#include "aarand/aarand.hpp"
#include "powerit/powerit.hpp"

#include "Initialize.hpp"
#include "compute_centroids.hpp"

/**
 * @file InitializeVariancePartition.hpp
 *
 * @brief Class for k-means initialization with PCA partitioning.
 */
namespace kmeans {

/**
 * @brief Options for `InitializeVariancePartition`.
 */
struct InitializeVariancePartitionOptions {
    /**
     * Size adjustment value, check out `InitializeVariancePartition` for more details.
     * This value should lie in `[0, 1]`.
     */
    double size_adjustment = 1;

    /**
     * @cond
     */
    // Only keeping these around for back-compatibility.
    powerit::Options power_iteration_options;

    uint64_t seed = 6523u;
    /**
     * @endcond
     */
};

/**
 * @cond
 */
namespace InitializeVariancePartition_internal {

template<class Dim_, typename Value_, typename Float_>
void compute_welford(Dim_ ndim, const Value_* dptr, Float_* center, Float_* dim_ss, Float_ count) {
    for (Dim_ j = 0; j < ndim; ++j) {
        Float_ val = dptr[j];
        auto cur_center = center[j];
        Float_ delta = val - cur_center;
        Float_ new_center = cur_center + delta / count;
        center[j] = new_center;
        dim_ss[j] += (val - new_center) * delta;
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Implements the PCA partitioning method of Su and Dy (2007).
 *
 * This approach involves the selection of starting points via iterative partitioning based on principal components analysis.
 * The aim is to obtain well-separated starting points for refinement with algorithms like Hartigan-Wong or Lloyd.
 * This is achieved by selecting the most dispersed cluster for further partitioning.
 *
 * We start from a single cluster containing all points.
 * At each iteration, we select the cluster with the largest within-cluster sum of squares (WCSS);
 * we identify the first principal component within that cluster;
 * and we split the cluster at its center along that axis to obtain two new clusters.
 * This is repeated until the desired number of clusters is obtained, and the centers and cluster identifiers are then reported.
 *
 * The original algorithm favors selection and partitioning of the largest cluster, which has the greatest chance of having the highest WCSS.
 * For more fine-grained control, we modify the procedure to adjust the effective number of observations contributing to the WCSS.
 * Specifically, we choose the cluster to partition based on the product of $N$ and the mean squared difference within each cluster,
 * where $N$ is the cluster size raised to some user-specified power (i.e., the "size adjustment") between 0 and 1.
 * An adjustment of 1 recapitulates the original algorithm, while smaller values of the size adjustment will reduce the preference towards larger clusters.
 * A value of zero means that the cluster size is completely ignored, though this seems unwise as it causes excessive splitting of small clusters with unstable WCSS.
 *
 * This method is not completely deterministic as a randomization step is used in the PCA.
 * Nonetheless, the stochasticity is likely to have a much smaller effect than in the other initialization methods.
 *
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 *
 * @see
 * Su, T. and Dy, J. G. (2007).
 * In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
 * _Intelligent Data Analysis_ 11, 319-338.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class InitializeVariancePartition : public Initialize<Matrix_, Cluster_, Float_> {
public:
    /**
     * @param options Options for PCA partitioning.
     */
    InitializeVariancePartition(InitializeVariancePartitionOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializeVariancePartition() = default;

private:
    InitializeVariancePartitionOptions my_options;

public:
    /**
     * @return Options for PCA partitioning,
     * to be modified prior to calling `run()`.
     */
    InitializeVariancePartitionOptions& get_options() {
        return my_options;
    }

public:
    Cluster_ run(const Matrix_& data, Cluster_ ncenters, Float_* centers) const {
        auto nobs = data.num_observations();
        auto ndim = data.num_dimensions();
        if (nobs == 0 || ndim == 0) {
            return 0;
        }

        std::vector<std::vector<typename Matrix_::index_type> > assignments(ncenters);
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);

        std::vector<std::vector<Float_> > dim_ss(ncenters);
        {
            auto& cur_ss = dim_ss[0]
            cur_ss.resize(ndim);
            std::fill_n(center, ndim, 0);
            auto matwork = data.create_workspace();
            for (decltype(nobs) i = 0; i < nobs; ++i) {
                auto dptr = data.get_observation(matwork);
                InitializeVariancePartition_internal::compute_welford(ndim, dptr, center, cur_ss, static_cast<Float_>(i + 1));
            }
        }

        std::priority_queue<std::pair<Float_, Cluster_> > highest;
        auto add_to_queue = [&](Cluster_ i) {
            const auto& cur_ss = dim_ss[i];
            Float_ sum_ss = std::accumulate(cur_ss.begin(), cur_ss.end(), static_cast<Float_>(0));
            sum_ss /= std::pow(assignments[i].size(), 1.0 - options.size_adjustment);
            highest.emplace_back(sum_ss, i);  
        };
        add_to_queue(0);

        std::vector<typename Matrix_::index_type> cur_assignments_copy;
        size_t long_ndim = ndim;

        for (Cluster_ cluster = 1; cluster < ncenters; ++cluster) {
            auto chosen = highest.top();
            if (chosen.first == 0) {
                return cluster; // no point continuing, we're at zero variance already.
            }
            highest.pop();

            auto& cur_ss = dim_ss[chosen.second];
            auto& next_ss = dim_ss[cluster];
            auto* cur_center = centers + static_cast<size_t>(chosen.second) * long_ndim; // cast to size_t to avoid overflow issues.
            auto* next_center = centers + static_cast<size_t>(cluster) * long_ndim;

            auto& cur_assignments = assignments[chosen.second];
            auto& next_assignments = assignments[cluster];
            cur_assignments_copy.clear();

            size_t top_dim = std::max_element(cur_ss.begin(), cur_ss.end()) - cur_ss.begin();
            auto top_center = cur_center[top_dim];
            std::fill_n(cur_center, ndim, 0);
            std::fill(cur_ss.begin(), cur_ss.end(), 0);

            size_t num_in_cluster = cur_assignments.size();
            auto work = data.create_workspace(cur_assignments.data(), num_in_cluster);
            for (auto i : cur_assignments) {
                auto dptr = data.get_observation(work);
                if (dptr[top_dim] < top_center) {
                    cur_assignments_copy.push_back(i);
                    InitializeVariancePartition_internal::compute_welford(ndim, dptr, cur_center, cur_ss, static_cast<Float_>(cur_assignments_copy.size()));
                } else {
                    next_assignments.push_back(i);
                    InitializeVariancePartition_internal::compute_welford(ndim, dptr, next_center, next_ss, static_cast<Float_>(next_assignments.size()));
                }
            }

            // If one or the other is empty, then this entire procedure short
            // circuits as all future iterations will just re-select this
            // cluster (which won't get partitioned properly anyway). In the
            // bigger picture, the quick exit out of the iterations is correct
            // as we should only fail to partition in this manner if all points
            // within each remaining cluster are identical.
            if (new_assignments.empty() || cur_assignments_copy.empty()) {
                return cluster;
            }

            cur_assignments.swap(cur_assignments_copy);
            add_to_queue(chosen.second);
            add_to_queue(cluster);
        }

        return ncenters;
    }
};

}

#endif
