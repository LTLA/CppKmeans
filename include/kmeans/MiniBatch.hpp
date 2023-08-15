#ifndef KMEANS_MINIBATCH_HPP
#define KMEANS_MINIBATCH_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <random>

#include "Base.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "compute_wcss.hpp"
#include "is_edge_case.hpp"
#include "random.hpp"

/**
 * @file MiniBatch.hpp
 *
 * @brief Implements the mini-batch algorithm for k-means clustering.
 */

namespace kmeans {

/**
 * @brief Implements the mini-batch algorithm for k-means clustering.
 *
 * The mini-batch approach is similar to Lloyd's algorithm in that it runs through a set of observations, assigns each to the closest centroid, updates the centroids and repeats.
 * The key difference is that each iteration is performed with a random subset of observations (i.e., a "mini-batch"), instead of the full set of observations.
 * This reduces computational time at the cost of some solution quality.
 * In theory, it can also reduce memory usage, though this is not particularly relevant here as we are already assuming that the data can be fully stored in memory.
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
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class MiniBatch : public Refine<DATA_t, CLUSTER_t, INDEX_t> {
public:
    /** 
     * @brief Default parameter values for `MiniBatch`.
     */
    struct Defaults {
        /** 
         * See `set_max_iterations()` for more details.
         */
        static constexpr int max_iterations = 100;

        /** 
         * See `set_batch_size()` for more details.
         */
        static constexpr INDEX_t batch_size = 500;

        /** 
         * See `set_max_change_proportion()` for more details.
         */
        static constexpr double max_change_proportion = 0.01;

        /** 
         * See `set_convergence_history()` for more details.
         */
        static constexpr int convergence_history = 10;

        /** 
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 1234567890;

        /** 
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };

private:
    int maxiter = Defaults::max_iterations;

    INDEX_t batch_size = Defaults::batch_size;

    int history = Defaults::convergence_history;

    double max_change = Defaults::max_change_proportion;

    uint64_t seed = Defaults::seed;

    int nthreads = Defaults::num_threads;
public:
    /**
     * @param i Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_max_iterations(int i = Defaults::max_iterations) {
        maxiter = i;
        return *this;
    }

    /**
     * @param s Number of observations in the mini-batch.
     * Larger numbers improve quality at the cost of computational time.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_batch_size(INDEX_t s = Defaults::batch_size) {
        batch_size = s;
        return *this;
    }

    /**
     * @param p Maximum proportion of observations in each cluster that can be reassigned when checking for convergence (i.e., \f$p\f$).
     * Lower values improve the quality of the result at the cost of computational time.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_max_change_proportion(double p = Defaults::max_change_proportion) {
        max_change = p;
        return *this;
    }

    /**
     * @param h Number of iterations to remember when checking for convergence (i.e., \f$h\f$).
     * Larger values improve the quality of the result at the cost of computational time.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_convergence_history(int h = Defaults::convergence_history) {
        history = h;
        return *this;
    }

    /** 
     * @param s Seed to use for PRNG when sampling observations to use in each mini-batch.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `MiniBatch` object.
     */
    MiniBatch& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On input, this should contain the initial centroid locations for each cluster.
     * Data should be stored in column-major order.
     * On output, this will contain the final centroid locations for each cluster.
     * @param[out] clusters Pointer to an array of length `nobs`.
     * On output, this will contain the cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * If `ncenters > nobs`, only the first `nobs` columns of the `centers` array will be filled.
     */
    Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        if (is_edge_case(nobs, ncenters)) {
            return process_edge_case(ndim, nobs, data, ncenters, centers, clusters);
        }

        int iter = 0, status = 0;
        std::vector<INDEX_t> total_sampled(ncenters);
        std::vector<CLUSTER_t> previous(nobs);
        std::vector<INDEX_t> last_changed(ncenters), last_sampled(ncenters);

        auto actual_batch_size = std::min(batch_size, nobs);
        std::mt19937_64 eng(seed);

        for (iter = 1; iter <= maxiter; ++iter) {
            auto chosen = sample_without_replacement(nobs, actual_batch_size, eng);
            if (iter > 1) {
                for (auto o : chosen) {
                    previous[o] = clusters[o];
                }
            }

            QuickSearch<DATA_t, CLUSTER_t> index(ndim, ncenters, centers);
            size_t nchosen = chosen.size();

#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads)
#endif
            for (size_t i = 0; i < nchosen; ++i) {
#else
            KMEANS_CUSTOM_PARALLEL(nchosen, [&](size_t first, size_t last) -> void {
            for (size_t i = first; i < last; ++i) {
#endif
                clusters[chosen[i]] = index.find(data + chosen[i] * ndim);
#ifndef KMEANS_CUSTOM_PARALLEL            
            }
#else
            }
            }, nthreads);
#endif        

            // Updating the means for each cluster.
            for (auto o : chosen) {
                const auto c = clusters[o];
                auto& n = total_sampled[c];
                ++n;

                auto ocopy = data + o * ndim;
                auto ccopy = centers + c * ndim;
                for (int d = 0; d < ndim; ++d, ++ocopy, ++ccopy) {
                    (*ccopy) += (*ocopy - *ccopy)/n;
                }
            }

            // Checking for updates.
            if (iter != 1) {
                for (auto o : chosen) {
                    ++(last_sampled[previous[o]]);
                    if (previous[o] != clusters[o]) {
                        ++(last_sampled[clusters[o]]);
                        ++(last_changed[previous[o]]);
                        ++(last_changed[clusters[o]]);
                    }
                }

                if (iter % history == 1) {
                    bool too_many_changes = false;
                    for (CLUSTER_t c = 0; c < ncenters; ++c) {
                        if (static_cast<double>(last_changed[c]) >= last_sampled[c] * max_change) {
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

        if (iter == maxiter + 1) {
            status = 2;
        }

        // Run through all observations to make sure they have the latest cluster assignments.
        QuickSearch<DATA_t, CLUSTER_t> index(ndim, ncenters, centers);

#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
#endif
        for (INDEX_t o = 0; o < nobs; ++o) {
#else
        KMEANS_CUSTOM_PARALLEL(nobs, [&](INDEX_t first, INDEX_t last) -> void {
        for (INDEX_t o = first; o < last; ++o) {
#endif
            clusters[o] = index.find(data + o * ndim);
#ifndef KMEANS_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        std::fill(total_sampled.begin(), total_sampled.end(), 0);
        for (INDEX_t o = 0; o < nobs; ++o) {
            ++total_sampled[clusters[o]];
        }

        for (auto c : total_sampled) {
            if (c == 0) {
                status = 1;
                break;
            }
        }

        compute_centroids(ndim, nobs, data, ncenters, centers, clusters, total_sampled);

        return Details<DATA_t, INDEX_t>(
            std::move(total_sampled),
            compute_wcss(ndim, nobs, data, ncenters, centers, clusters),
            iter, 
            status
        );
    }
};

}

#endif
