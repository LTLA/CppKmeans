#ifndef KMEANS_REFINE_MINIBATCH_HPP
#define KMEANS_REFINE_MINIBATCH_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <random>
#include <type_traits>

#include "aarand/aarand.hpp"

#include "Refine.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "is_edge_case.hpp"
#include "parallelize.hpp"

/**
 * @file RefineMiniBatch.hpp
 *
 * @brief Implements the mini-batch algorithm for k-means clustering.
 */

namespace kmeans {

/** 
 * @brief Options for `RefineMiniBatch` construction.
 */
struct RefineMiniBatchOptions {
    /** 
     * Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     */
    int max_iterations = 100;

    /** 
     * Number of observations in the mini-batch.
     * Larger numbers improve quality at the cost of computational time and memory.
     */
    int batch_size = 500;

    /** 
     * Maximum proportion of observations in each cluster that can be reassigned when checking for convergence (i.e., \f$p\f$ in the documentation for `RefineMiniBatch`).
     * Lower values improve the quality of the result at the cost of computational time.
     */
    double max_change_proportion = 0.01;

    /** 
     * Number of iterations to remember when checking for convergence (i.e., \f$h\f$ in the documentation for `RefineMiniBatch`).
     * Larger values improve the quality of the result at the cost of computational time.
     */
    int convergence_history = 10;

    /** 
     * Seed to use for the PRNG when sampling observations to use in each mini-batch.
     */
    uint64_t seed = 1234567890u;

    /** 
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Implements the mini-batch algorithm for k-means clustering.
 *
 * The mini-batch approach is similar to Lloyd's algorithm in that it runs through a set of observations, assigns each to the closest centroid, updates the centroids and repeats.
 * The key difference is that each iteration is performed with a random subset of observations (i.e., a "mini-batch"), instead of the full set of observations.
 * This reduces computational time and memory usage at the cost of some solution quality.
 * 
 * The update procedure for a cluster's centroid involves adjusting the coordinates by the assigned observations in the mini-batch.
 * The resulting vector can be interpreted as the mean of all observations that have ever been sampled (possibly multiple times) to that cluster.
 * Thus, the magnitude of the updates will decrease in later iterations as the relative effect of newly sampled points is reduced.
 * This ensures that the centroids will stabilize at a sufficiently large number of iterations.
 *
 * We may stop the algorithm before the maximum number of iterations if only a few observations are reassigned at each iteration. 
 * Specifically, every \f$h\f$ iterations, we compute the proportion of sampled observations for each cluster in the past \f$h\f$ mini-batches that were reassigned to/from that cluster.
 * If this proportion is less than some threshold \f$p\f$ for all clusters, we consider that the algorithm has converged.
 * 
 * In the `Details::status` returned by `run()`, the status code is either 0 (success) or 2 (maximum iterations reached without convergence).
 * Previous versions of the library would report a status code of 1 upon encountering an empty cluster, but these are now just ignored.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the data.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Type for the input data matrix.
 * This should satisfy the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, typename Matrix_ = Matrix<Index_, Data_> >
class RefineMiniBatch : public Refine<Index_, Data_, Cluster_, Float_, Matrix_> {
public:
    /**
     * @param options Further options for the mini-batch algorithm.
     */
    RefineMiniBatch(RefineMiniBatchOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    RefineMiniBatch() = default;

public:
    /**
     * @return Options for mini-batch partitioning,
     * to be modified prior to calling `run()`.
     */
    RefineMiniBatchOptions& get_options() {
        return my_options;
    }

private:
    RefineMiniBatchOptions my_options;

public:
    /**
     * @cond
     */
    Details<Index_> run(const Matrix_& data, Cluster_ ncenters, Float_* centers, Cluster_* clusters) const {
        Index_ nobs = data.num_observations();
        if (internal::is_edge_case(nobs, ncenters)) {
            return internal::process_edge_case(data, ncenters, centers, clusters);
        }

        int iter = 0, status = 0;
        std::vector<uint64_t> total_sampled(ncenters); // holds the number of sampled observations across iterations, so we need a large integer.
        std::vector<Cluster_> previous(nobs);
        std::vector<uint64_t> last_changed(ncenters), last_sampled(ncenters); // holds the number of sampled/changed observation for the last few iterations.

        Index_ actual_batch_size = nobs;
        typedef typename std::conditional<std::is_signed<Index_>::value, int, unsigned int>::type SafeCompInt; // waiting for C++20's comparison functions...
        if (static_cast<SafeCompInt>(actual_batch_size) > my_options.batch_size) {
            actual_batch_size = my_options.batch_size;
        }
        std::vector<Index_> chosen(actual_batch_size);
        std::mt19937_64 eng(my_options.seed);

        size_t ndim = data.num_dimensions();
        internal::QuickSearch<Float_, Cluster_> index;

        for (iter = 1; iter <= my_options.max_iterations; ++iter) {
            aarand::sample(nobs, actual_batch_size, chosen.data(), eng);
            if (iter > 1) {
                for (auto o : chosen) {
                    previous[o] = clusters[o];
                }
            }

            index.reset(ndim, ncenters, centers);
            parallelize(my_options.num_threads, actual_batch_size, [&](int, Index_ start, Index_ length) -> void {
                auto work = data.new_extractor(chosen.data() + start, length);
                for (Index_ s = start, end = start + length; s < end; ++s) {
                    auto ptr = work->get_observation();
                    clusters[chosen[s]] = index.find(ptr);
                }
            });

            // Updating the means for each cluster.
            auto work = data.new_extractor(chosen.data(), actual_batch_size);
            for (auto o : chosen) {
                const auto c = clusters[o];
                auto& n = total_sampled[c];
                ++n;

                Float_ mult = static_cast<Float_>(1)/static_cast<Float_>(n);
                auto ccopy = centers + static_cast<size_t>(c) * ndim; // cast to size_t to avoid overflow.
                auto ocopy = work->get_observation();

                for (size_t d = 0; d < ndim; ++d) {
                    ccopy[d] += (static_cast<Float_>(ocopy[d]) - ccopy[d]) * mult; // cast to ensure consistent precision regardless of Matrix_::data_type.
                }
            }

            // Checking for updates.
            if (iter != 1) {
                for (auto o : chosen) {
                    auto p = previous[o];
                    ++(last_sampled[p]);
                    auto c = clusters[o];
                    if (p != c) {
                        ++(last_sampled[c]);
                        ++(last_changed[p]);
                        ++(last_changed[c]);
                    }
                }

                if (iter % my_options.convergence_history == 1) {
                    bool too_many_changes = false;
                    for (Cluster_ c = 0; c < ncenters; ++c) {
                        if (static_cast<double>(last_changed[c]) >= static_cast<double>(last_sampled[c]) * my_options.max_change_proportion) {
                            too_many_changes = true;
                            break;
                        }
                    }

                    if (!too_many_changes) {
                        break;
                    }
                    std::fill(last_sampled.begin(), last_sampled.end(), 0);
                    std::fill(last_changed.begin(), last_changed.end(), 0);
                }
            }
        }

        if (iter == my_options.max_iterations + 1) {
            status = 2;
        }

        // Run through all observations to make sure they have the latest cluster assignments.
        index.reset(ndim, ncenters, centers);
        parallelize(my_options.num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
            auto work = data.new_extractor(start, length);
            for (Index_ s = start, end = start + length; s < end; ++s) {
                auto ptr = work->get_observation();
                clusters[s] = index.find(ptr);
            }
        });

        std::vector<Index_> cluster_sizes(ncenters);
        for (Index_ o = 0; o < nobs; ++o) {
            ++cluster_sizes[clusters[o]];
        }

        internal::compute_centroids(data, ncenters, centers, clusters, cluster_sizes);
        return Details<Index_>(std::move(cluster_sizes), iter, status);
    }
    /**
     * @endcond
     */
};

}

#endif
