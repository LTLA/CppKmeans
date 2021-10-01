#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include "Base.hpp"
#include "HartiganWong.hpp"
#include "initialization.hpp"
#include "Details.hpp"
#include <random>

/** 
 * @file Kmeans.hpp
 *
 * @brief Implements the full k-means clustering procedure.
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
         * See `set_weighted()` for more details.
         */
        static constexpr bool weighted = true;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 5489u;
    };

private:
    bool weighted_init = Defaults::weighted; 
    uint64_t seed = Defaults::seed;

public:
    /** 
     * @param w Should we use weighted initialization (see `weighted_initialization()`)?
     * If `false`, simple initialization is used instead (see `simple_initialization()`).
     *
     * @return A reference to this `Kmeans` object.
     */
    Kmeans& set_weighted(bool w = true) {
        weighted_init = w;
        return *this;
    }

    /** 
     * @param s Seed to use for PRNG.
     * Defaults to default seed for the `std::mt19937_64` constructor.
     *
     * @return A reference to this `Kmeans` object.
     */
    Kmeans& set_seed(uint64_t s = 5489u) {
        seed = s;
        return *this;
    }
private:
    CLUSTER_t initialize(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers) {
        std::vector<INDEX_t> chosen;
        std::mt19937_64 rng(seed);
        if (weighted_init) {
            chosen = weighted_initialization(ndim, nobs, data, ncenters, rng);
        } else {
            chosen = simple_initialization(nobs, ncenters, rng);
        }

        for (auto c : chosen) {
            std::copy(data + c * ndim, data + (c + 1) * ndim, centers);
            centers += ndim;
        }

        return chosen.size(); // this may be less than ncenters, depending on the initializer.
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
     * On output, this will contain the (0-indexed) cluster assignment for each observation.
     * @param algorithm Pointer to a `Base` object containing the desired k-means algorithm.
     * If `NULL`, this defaults to a default `HartiganWong` instance.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * Note that the actual number of clusters may be less than `ncenters` in pathological cases - 
     * check the length of `Details::sizes` and the value of `Details::status`.
     */
    Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters,  Base<DATA_t, CLUSTER_t, INDEX_t>* algorithm = NULL) {
        ncenters = initialize(ndim, nobs, data, ncenters, centers); 

        if (algorithm == NULL) {
            HartiganWong<DATA_t, CLUSTER_t, INDEX_t> hw;
            return hw.run(ndim, nobs, data, ncenters, centers, clusters);
        } else {
            return algorithm->run(ndim, nobs, data, ncenters, centers, clusters);
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
     * @param algorithm Pointer to a `Base` object containing the desired k-means algorithm.
     * If `NULL`, this defaults to a default `HartiganWong` instance.
     *
     * @return `centers` and `clusters` are filled, and a `Results` object is returned containing clustering statistics.
     * See `run()` for more details.
     */
    Results run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, Base<DATA_t, CLUSTER_t, INDEX_t>* algorithm = NULL) {
        Results output(ndim, nobs, ncenters);
        ncenters = initialize(ndim, nobs, data, ncenters, output.centers.data()); 
        output.details = run(ndim, nobs, data, ncenters, output.centers.data(), output.clusters.data(), algorithm);
        return output;
    }
};

}

#endif
