#ifndef KMEANS_INITIALIZE_PCA_PARTITION_HPP
#define KMEANS_INITIALIZE_PCA_PARTITION_HPP

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "aarand/aarand.hpp"
#include "powerit/PowerIterations.hpp"
#include "Base.hpp"

/**
 * @file InitializePcaPartition.hpp
 *
 * @brief Class for k-means initialization with PCA partitioning.
 */
namespace kmeans {

/**
 * @brief Options for `InitializePcaPartition`.
 */
struct InitializePcaPartitionOptions {
    /**
     * Options to use for the power iterations.
     */
    powerit::Options power_iteration_options;

    /**
     * Size adjustment value, check out `InitializePcaPartition` for more details.
     * This value should lie in `[0, 1]`.
     */
    double size_adjustment = 1;

    /**
     * Random seed to use to construct the PRNG for the power iterations.
     */
    static constexpr uint64_t seed = 6523u;
};

/**
 * @cond
 */
namespace InitializePcaPartitionsOptions_internal {

template<typename Center_>
struct PowerWorkspace {
    std::vector<Center_> pc;
    std::vector<Center_> delta;
    std::vector<Center_> cov;
};

template<class Matrix_, typename Center_, class Engine_>
std::vector<Data_> compute_pc1(
    const Matrix_& data, 
    const std::vector<typename Matrix_::index_type>& chosen, 
    const Center_* center, 
    Engine_& eng, 
    PowerWorkspace<Matrix_, Center_>& work,
    int nthreads,
    const powerit::Options& power_opts)
{
    auto ndim = data.num_observations();
    work.pc.resize(ndim);
    work.delta.resize(ndim);
    size_t long_ndim = ndim;
    work.cov.resize(long_ndim * long_ndim);

    // Computing the lower triangle of the covariance matrix. 
    // TODO: parallelize this bad boy by splitting into blocks, computing the
    // covariance within each block in each thread, and then adding them all
    // together. We need to use blocks to guarantee we get the same results
    // with and without parallelization.
    auto work = data.get_workspace(chosen.data(), chosen.size());
    for (size_t i = 0, end = chosen.size(); i < end; ++i) {
        auto dptr = data.get_observation(work);

        for (int j = 0; j < ndim; ++j) {
            delta[j] = dptr[j] - center[j];
        }

        size_t offset = 0;
        for (decltype(ndim) j = 0; j < ndim; ++j, offset += long_ndim) {
            for (decltype(ndim) k = 0; k <= j; ++k) {
                cov[offset + static_cast<size_t>(k)] += delta[j] * delta[k];
            }
        }
    }

    // Filling in the other side of the matrix, to enable cache-efficient multiplication.
    size_t src_offset = 0;
    for (int j = 0; j < ndim; ++j, src_offset += long_ndim) {
        size_t dest_offset = j;
        size_t src_offset_copy = src_offset;
        for (int k = j + 1; k < ndim; ++k, dest_offset += long_dim, ++src_offset_copy) {
            cov[src_offset_copy] = cov[dest_offset];
        }
    }

    powerit::compute(ndim, cov.data(), work.pc.data(), eng, power_opts);
    return output;
} 

template<class Matrix_, typename Center_>
void compute_center(const Matrix_& data, Data_* center) {
    auto ndim = data.num_dimensions();
    std::fill_n(center, ndim, 0);
    auto nobs = data.num_observations();
    auto work = data.create_workspace(0, nobs);

    for (decltype(nobs) i = 0; i < nobs; ++i) {
        auto dptr = data.fetch_observation(work);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            center[d] += dptr[d];
        }
    }

    for (decltype(ndim) d = 0; d < ndim; ++d) {
        center[d] /= nobs;
    }
}

template<class Matrix_, typename Center_>
Center_ update_mrse(const Matrix_& data, const std::vector<typename Matrix_::index_type>& chosen, Center_* center, std::vector<Center_>& work_var) {
    std::fill_n(center, ndim, 0);
    std::fill_n(work_var.begin(), ndim, 0);

    auto ndim = data.num_dimensions();
    auto work = data.create_workspace(chosen.data(), chosen.size());

    // Using Welford's method so we only have to make one pass through the matrix,
    // instead of doing two passes to compute the mean and then the variance.
    for (size_t i = 0, end = chosen.size(); i < end; ++i) {
        auto dptr = work.fetch_observation(work);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            Data_ delta = dptr[d] - center[d];
            center[d] += delta / (i + 1);
            work_var[d] += delta * (dptr[d] - center[d]);
        }
    }

    return std::accumulate(work_var.begin(), work_var.end(), static_cast<Center_>(0)) / chosen.size();
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
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster index.
 * @tparam Index_ Integer type for the observation index.
 *
 * @see
 * Su, T. and Dy, J. G. (2007).
 * In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
 * _Intelligent Data Analysis_ 11, 319-338.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
class InitializePcaPartition : public Initialize<Matrix_, Cluster_, Center_> {
public:
    /**
     * @param options Options for PCA partitioning.
     */
    InitializePcaPartition(InitializePcaPartition<Data_> options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializePcaPartition() = default;

private:
    InitializePcaPartitionOptions my_options;

public:
    Cluster_ run(const Matrix_& data, Cluster_ ncenters, Center_* centers) {
        auto nobs = matrix.num_observations();
        if (nobs == 0) {
            return 0;
        }

        std::mt19937_64 rng(seed);
        std::priority_queue<std::pair<Data_, Cluster_> > mrse;
        std::vector<std::vector<Index_> > assignments(ncenters);

        auto ndim = matrix.num_dimensions();
        std::vector<Center_> work_var(ndim);

        // Setting up the zero'th cluster. (No need to actually compute the
        // MRSE at this point, as there's nothing to compare it to.)
        compute_center(matrix, centers);
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);
        std::fill_n(clusters, nobs, 0);

        for (Cluster_ cluster = 1; cluster < ncenters; ++cluster) {
            Index_ worst_cluster = 0;
            if (worst.size()) {
                worst_cluster = mrse.top().second;
                mrse.pop();
            }

            // Extracting the principal component for this bad boy.
            auto worst_center = centers + worst_cluster * ndim;
            auto& worst_assignments = assignments[worst_cluster];
            auto pc1 = compute_pc1(data, worst_assignments, worst_center, rng);

            // Projecting all points in this cluster along PC1. The center lies
            // at zero, so everything positive (on one side of the hyperplane
            // orthogonal to PC1 and passing through the center) gets bumped to
            // the next cluster.
            std::vector<Index_>& new_assignments = assignments[cluster];
            std::vector<Index_> worst_assignments2;

            size_t num_in_cluster = worst_assignments.size();
            auto work = data.create_workspace(worst_assignments.data(), num_in_cluster);
            for (size_t i = 0; i < num_in_cluster; ++i) {
                auto dptr = data.fetch_observation(work);

                Center_ proj = 0;
                for (decltype(ndim) d = 0; d < ndim; ++d) {
                    proj += (dptr[d] - worst_center[d]) * pc1[d];
                }

                if (proj > 0) {
                    new_assignments.push_back(i);
                } else {
                    worst_assignments2.push_back(i);
                }
            }

            // If one or the other is empty, then this entire procedure short
            // circuits as all future iterations will just re-select this
            // cluster (which won't get partitioned properly anyway). In the
            // bigger picture, the quick exit out of the iterations is correct
            // as we should only fail to partition in this manner if all points
            // within each remaining cluster are identical.
            if (new_assignments.empty() || worst_assignments2.empty()) {
                return cluster;
            }

            for (auto i : new_assignments) {
                clusters[i] = cluster;
            }
            worst_assignments.swap(worst_assignments2);

            // Computing centers and MRSE.
            auto new_center = centers + static_cast<size_t>(cluster) * static_cast<size_t>(ndim); // cast to avoid overflow.
            mrse.emplace_back(update_mrse(data, new_assignments, new_center, work_var), cluster);
            mrse.emplace_back(update_mrse(data, worst_assignments, worst_center, work_var), worst_cluster);
        }

        return ncenters;
    }
};

}

#endif
