#ifndef KMEANS_INITIALIZE_VARIANCE_PARTITION_HPP
#define KMEANS_INITIALIZE_VARIANCE_PARTITION_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <cstdint>

#include "Initialize.hpp"
#include "Matrix.hpp"

/**
 * @file InitializeVariancePartition.hpp
 *
 * @brief k-means initialization with variance partitioning.
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
     * Whether to optimize the partition boundary to minimize the sum of sum of squares within each of the two child partitions.
     * If false, the partition boundary is simply defined as the mean.
     */
    bool optimize_partition = true;
};

/**
 * @cond
 */
namespace InitializeVariancePartition_internal {

template<typename Float_>
void compute_welford(Float_ val, Float_& center, Float_& ss, Float_ count) {
    auto cur_center = center;
    Float_ delta = val - cur_center;
    Float_ new_center = cur_center + delta / count;
    center = new_center;
    ss += (val - new_center) * delta;
}

template<typename Data_, typename Float_>
void compute_welford(size_t ndim, const Data_* dptr, Float_* center, Float_* dim_ss, Float_ count) {
    for (size_t d = 0; d < ndim; ++d) {
        compute_welford<Float_>(dptr[d], center[d], dim_ss[d], count);
    }
}

template<typename Matrix_, typename Index_, typename Float_>
Float_ optimize_partition(
    const Matrix_& data,
    const std::vector<Index_>& current,
    size_t top_dim,
    std::vector<Float_>& value_buffer,
    std::vector<Float_>& stat_buffer)
{
    /**
     * This function effectively implements a much more efficient
     * version of the following prototype code in R:
     *
     * a <- sort(c(rnorm(100, -1), rnorm(1000, 5)))
     * stuff <- numeric(length(a))
     * for (i in seq_along(a)) {
     *     mid <- a[i]
     *     left <- a[a<mid]
     *     right <- a[a>=mid]
     *     stuff[i] <- sum((left - mean(left))^2) + sum((right - mean(right))^2)
     * }
     * plot(a, stuff)
     */

    size_t N = current.size();
    auto work = data.new_extractor(current.data(), N);
    value_buffer.clear();
    for (size_t i = 0; i < N; ++i) {
        auto dptr = work->get_observation();
        value_buffer.push_back(dptr[top_dim]);
    }

    std::sort(value_buffer.begin(), value_buffer.end());

    stat_buffer.clear();
    stat_buffer.reserve(value_buffer.size() + 1);
    stat_buffer.push_back(0); // first and last entries of stat_buffer are 'nonsense' partitions, i.e., all points in one partition or the other.

    // Forward and backward iterations to get the sum of squares at every possible partition point.
    Float_ mean = 0, ss = 0, count = 1;
    for (auto val : value_buffer) {
        compute_welford<Float_>(val, mean, ss, count);
        stat_buffer.push_back(ss);
        ++count;
    }

    mean = 0, ss = 0, count = 1;
    auto sbIt = stat_buffer.rbegin() + 1; // skipping the last entry, as it's a nonsense partition.
    for (auto vIt = value_buffer.rbegin(), vEnd = value_buffer.rend(); vIt != vEnd; ++vIt) {
        compute_welford<Float_>(*vIt, mean, ss, count);
        *sbIt += ss;
        ++count;
        ++sbIt;
    }

    auto sbBegin = stat_buffer.begin(), sbEnd = stat_buffer.end();
    auto which_min = std::min_element(sbBegin, sbEnd) - sbBegin;
    if (which_min == 0) {
        // Should never get to this point as current.size() > 1,
        // but we'll just add this in for safety's sake.
        return value_buffer[0];
    } else {
        // stat_buffer[i] represents the SS when {0, 1, 2, ..., i-1} goes in
        // the left partition and {i, ..., N-1} goes in the right partition.
        // To avoid issues with ties, we use the average of the two edge
        // points as the partition boundary.
        return (value_buffer[which_min - 1] + value_buffer[which_min]) / 2;
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Implements the variance partitioning method of Su and Dy (2007).
 *
 * We start from a single cluster containing all points.
 * At each iteration, we select the cluster with the largest within-cluster sum of squares (WCSS);
 * we identify the dimension with the largest variance within that cluster;
 * and we split the cluster at its center along that axis to obtain two new clusters.
 * This is repeated until the desired number of clusters is obtained.
 * The idea is to deterministically partition the dataset so that the initial centers are evenly distributed along the axes of greatest variation.
 *
 * The original algorithm favors selection and partitioning of the largest cluster, which has the greatest chance of having the highest WCSS.
 * For more fine-grained control, we modify the procedure to adjust the effective number of observations contributing to the WCSS.
 * Specifically, we choose the cluster to partition based on the product of \f$N\f$ and the mean squared difference within each cluster,
 * where \f$N\f$ is the cluster size raised to some user-specified power (i.e., the "size adjustment") between 0 and 1.
 * An adjustment of 1 recapitulates the original algorithm, while smaller values of the size adjustment will reduce the preference towards larger clusters.
 * A value of zero means that the cluster size is completely ignored, though this seems unwise as it causes excessive splitting of small clusters with unstable WCSS.
 *
 * The original algorithm splits the cluster at the center (i.e., mean) along its most variable dimension.
 * We provide an improvement on this approach where the partition boundary is chosen to minimizes the sum of sum of squares of the two child partitions.
 * This often provides more appropriate partitions by considering the distribution of observations within the cluster, at the cost of some extra computation.
 * Users can switch back to the original approach by setting `InitializeVariancePartitionOptions::optimize_partition = false`.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
 *
 * @see
 * Su, T. and Dy, J. G. (2007).
 * In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
 * _Intelligent Data Analysis_ 11, 319-338.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class InitializeVariancePartition final : public Initialize<Index_, Data_, Cluster_, Float_, Matrix_> {
public:
    /**
     * @param options Options for variance partitioning.
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
     * @return Options for variance partitioning,
     * to be modified prior to calling `run()`.
     */
    InitializeVariancePartitionOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Cluster_ run(const Matrix_& data, Cluster_ ncenters, Float_* centers) const {
        auto nobs = data.num_observations();
        size_t ndim = data.num_dimensions();
        if (nobs == 0 || ndim == 0) {
            return 0;
        }

        std::vector<std::vector<Index_> > assignments(ncenters);
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);

        std::vector<std::vector<Float_> > dim_ss(ncenters);
        {
            auto& cur_ss = dim_ss[0];
            cur_ss.resize(ndim);
            std::fill_n(centers, ndim, 0);
            auto matwork = data.new_extractor(static_cast<Index_>(0), nobs);
            for (decltype(nobs) i = 0; i < nobs; ++i) {
                auto dptr = matwork->get_observation();
                InitializeVariancePartition_internal::compute_welford(ndim, dptr, centers, cur_ss.data(), static_cast<Float_>(i + 1));
            }
        }

        std::priority_queue<std::pair<Float_, Cluster_> > highest;
        auto add_to_queue = [&](Cluster_ i) -> void {
            const auto& cur_ss = dim_ss[i];
            Float_ sum_ss = std::accumulate(cur_ss.begin(), cur_ss.end(), static_cast<Float_>(0));

            // Instead of dividing by N and then remultiplying by pow(N, adjustment), we just
            // divide by pow(N, 1 - adjustment) to save some time and precision.
            sum_ss /= std::pow(assignments[i].size(), 1.0 - my_options.size_adjustment);

            highest.emplace(sum_ss, i);  
        };
        add_to_queue(0);

        std::vector<Index_> cur_assignments_copy;
        std::vector<Float_> opt_partition_values, opt_partition_stats;

        for (Cluster_ cluster = 1; cluster < ncenters; ++cluster) {
            auto chosen = highest.top();
            if (chosen.first == 0) {
                return cluster; // no point continuing, we're at zero variance already.
            }
            highest.pop();

            auto* cur_center = centers + static_cast<size_t>(chosen.second) * ndim; // cast to size_t to avoid overflow issues.
            auto& cur_ss = dim_ss[chosen.second];
            auto& cur_assignments = assignments[chosen.second];

            size_t top_dim = std::max_element(cur_ss.begin(), cur_ss.end()) - cur_ss.begin();
            Float_ partition_boundary;
            if (my_options.optimize_partition) {
                partition_boundary = InitializeVariancePartition_internal::optimize_partition(data, cur_assignments, top_dim, opt_partition_values, opt_partition_stats);
            } else {
                partition_boundary = cur_center[top_dim];
            }

            auto* next_center = centers + static_cast<size_t>(cluster) * ndim; // again, size_t to avoid overflow.
            std::fill_n(next_center, ndim, 0);
            auto& next_ss = dim_ss[cluster];
            next_ss.resize(ndim);
            auto& next_assignments = assignments[cluster];

            cur_assignments_copy.clear();
            std::fill_n(cur_center, ndim, 0);
            std::fill(cur_ss.begin(), cur_ss.end(), 0);
            auto work = data.new_extractor(cur_assignments.data(), cur_assignments.size());

            for (auto i : cur_assignments) {
                auto dptr = work->get_observation(); // make sure this is outside the if(), as it must always be called in each loop iteration to match 'cur_assignments' properly.
                if (dptr[top_dim] < partition_boundary) {
                    cur_assignments_copy.push_back(i);
                    InitializeVariancePartition_internal::compute_welford(ndim, dptr, cur_center, cur_ss.data(), static_cast<Float_>(cur_assignments_copy.size()));
                } else {
                    next_assignments.push_back(i);
                    InitializeVariancePartition_internal::compute_welford(ndim, dptr, next_center, next_ss.data(), static_cast<Float_>(next_assignments.size()));
                }
            }

            // If one or the other is empty, then this entire procedure short
            // circuits as all future iterations will just re-select this
            // cluster (which won't get partitioned properly anyway). In the
            // bigger picture, the quick exit out of the iterations is correct
            // as we should only fail to partition in this manner if all points
            // within each remaining cluster are identical.
            if (next_assignments.empty() || cur_assignments_copy.empty()) {
                return cluster;
            }

            cur_assignments.swap(cur_assignments_copy);
            add_to_queue(chosen.second);
            add_to_queue(cluster);
        }

        return ncenters;
    }
    /**
     * @endcond
     */
};

}

#endif
