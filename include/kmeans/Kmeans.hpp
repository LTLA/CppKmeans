#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include "Base.hpp"
#include "HartiganWong.hpp"
#include "InitializeKmeansPP.hpp"
#include "Details.hpp"
#include <random>

/** 
 * @file Kmeans.hpp
 *
 * @brief Implements the full k-means clustering procedure.
 */

/**
 * @namespace kmeans
 * @brief Namespace for k-means clustering.
 */
namespace kmeans {

/**
 * @brief Top-level class to run k-means clustering.
 *
 * k-means clustering aims to partition a dataset of `nobs` observations into `ncenters` clusters where `ncenters` is specified in advance.
 * Each observation is assigned to its closest cluster based on the distance to the cluster centroids.
 * The cluster centroids themselves are chosen to minimize the sum of squared Euclidean distances from each observation to its assigned cluster.
 * This procedure involves some heuristics to choose a good initial set of centroids (see `weighted_initialization()` for details) 
 * and to converge to a local minimum (see `HartiganWong`, `Lloyd` or `MiniBatch` for details).
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 * This should have a maximum positive value that is at least 50 times greater than the maximum expected number of observations.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class Kmeans {
public:
    /** 
     * @brief Default parameter values for `Kmeans`.
     */
    struct Defaults {
        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 5489u;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };

private:
    uint64_t seed = Defaults::seed;
    int nthreads = Defaults::num_threads;

public:
    /** 
     * @param s Seed to use for PRNG.
     * Defaults to default seed for the `std::mt19937_64` constructor.
     *
     * @return A reference to this `Kmeans` object.
     *
     * This seed is only used for the default `initializer` instance in `run()`.
     * Otherwise, if an `initializer` is explicitly passed to `run()`, its seed is respected.
     */
    Kmeans& set_seed(uint64_t s = 5489u) {
        seed = s;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `Kmeans` object.
     *
     * This setting is only used for the default `refiner` and `initializer` instances in `run()`.
     * Otherwise, if an `initializer` or `refiner` is explicitly passed to `run()`, the number of threads specified in the instance is respected.
     */
    Kmeans& set_num_threads(int n = Defaults::num_threads) {
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
     * On input, this should contain the initial centroid locations for each cluster if `set_initialization_method()` is `NONE`, otherwise it is ignored.
     * On output, this will contain the final centroid locations for each cluster.
     * Data should be stored in column-major order.
     * @param[in, out] clusters Pointer to an array of length `nobs`.
     * On input, this should contain the identity of the closest cluster for each observation if `set_initialization_method()` is `REINIT_PRECOMPUTED`, otherwise it is ignored.
     * On output, this will contain the (0-indexed) cluster assignment for each observation.
     * @param initializer Pointer to a `Initialize` object containing the desired k-means initialization method, e.g., `InitializeNone`, `InitializeRandom`, `InitializeKmeansPP`.
     * If `NULL`, this defaults to a default-constructed `InitializeKmeansPP` instance.
     * @param refiner Pointer to a `Refine` object containing the desired k-means refinement algorithm, e.g., `HartiganWong`, `Lloyd`, `MiniBatch`.
     * If `NULL`, this defaults to a default-constructed `HartiganWong` instance.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * Note that the actual number of clusters may be less than `ncenters` in pathological cases - 
     * check the length of `Details::sizes` and the value of `Details::status`.
     */
    Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters, 
        Initialize<DATA_t, CLUSTER_t, INDEX_t>* initializer = NULL, Refine<DATA_t, CLUSTER_t, INDEX_t>* refiner = NULL)
    {
        if (initializer == NULL) {
            InitializeKmeansPP<DATA_t, CLUSTER_t, INDEX_t> init;
            init.set_seed(seed).set_num_threads(nthreads);
            ncenters = init.run(ndim, nobs, data, ncenters, centers, clusters); 
        } else {
            ncenters = initializer->run(ndim, nobs, data, ncenters, centers, clusters); 
        }

        if (refiner == NULL) {
            HartiganWong<DATA_t, CLUSTER_t, INDEX_t> hw;
            hw.set_num_threads(nthreads);
            return hw.run(ndim, nobs, data, ncenters, centers, clusters);
        } else {
            return refiner->run(ndim, nobs, data, ncenters, centers, clusters);
        }
    }

public:
    /**
     * @brief Full statistics from k-means clustering.
     */
    struct Results {
        /**
         * @cond
         */
        Results(int ndim, INDEX_t nobs, CLUSTER_t ncenters) : centers(ndim * ncenters), clusters(nobs) {}

        Results() {}
        /**
         * @endcond
         */

        /**
         * A column-major `ndim`-by-`ncenters` array containing per-cluster centroid coordinates.
         */
        std::vector<DATA_t> centers;

        /**
         * An array of length `nobs` containing 0-indexed cluster assignments for each observation.
         */
        std::vector<CLUSTER_t> clusters;

        /**
         * Further details from the chosen k-means algorithm.
         */
        Details<DATA_t, INDEX_t> details;
    };

    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param initializer Pointer to a `Initialize` object containing the desired k-means initialization method.
     * If `NULL`, this defaults to a default-constructed `InitializeKmeansPP` instance.
     * @param refiner Pointer to a `Refine` object containing the desired k-means refinement algorithm.
     * If `NULL`, this defaults to a default-constructed `HartiganWong` instance.
     *
     * @return `centers` and `clusters` are filled, and a `Results` object is returned containing clustering statistics.
     * See `run()` for more details.
     *
     * For this method, it would be unwise to initialize with any methods that use information from the existing cluster centers. 
     * We suggest using only `InitializeRandom` or InitializeKmeansPP` here.
     */
    Results run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, 
        Initialize<DATA_t, CLUSTER_t, INDEX_t>* initializer = NULL, Refine<DATA_t, CLUSTER_t, INDEX_t>* refiner = NULL)
    {
        Results output(ndim, nobs, ncenters);
        output.details = run(ndim, nobs, data, ncenters, output.centers.data(), output.clusters.data(), initializer, refiner);
        return output;
    }
};

}

#endif
