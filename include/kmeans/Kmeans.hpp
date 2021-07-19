#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include "HartiganWong.hpp"
#include "initialization.hpp"
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
 * and to converge to a local minimum (see `HartiganWong` for details).
 */
class Kmeans {
    bool weighted_init = true;
    uint64_t seed = 5489u;

    HartiganWong hw;

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
    Kmeans& set_seed(bool s = 5489u) {
        seed = s;
        return *this;
    }

    /** 
     * Set the maximum number of iterations to the default in `HartiganWong::set_max_iterations()`.
     *
     * @return A reference to this `Kmeans` object.
     */
    Kmeans& set_max_iterations() {
        hw.set_max_iterations();
        return *this;
    }

    /** 
     * @param m Maximum number of iterations to use in the Hartigan-Wong algorithm.
     *
     * @return A reference to this `Kmeans` object.
     */
    Kmeans& set_max_iterations(int m) {
        hw.set_max_iterations(m);
        return *this;
    }

private:
    int initialize(int ndim, int nobs, const double* data, int ncenters, double* centers) {
        std::vector<int> chosen;
        std::mt19937_64 rng(seed);
        if (weighted_init) {
            chosen = weighted_initialization(ndim, nobs, data, ncenters, rng);
        } else {
            chosen = simple_initialization(ndim, nobs, data, ncenters, rng);
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
     * Om output, this will contain the (0-indexed) cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled, and a `HartiganWong::Details` object is returned containing clustering statistics.
     * Note that the actual number of clusters may be less than `ncenters` in pathological cases - 
     * check the length of `HartiganWong::Details::sizes` and the value of `HartiganWong::Details::status`.
     */
    HartiganWong::Details run(int ndim, int nobs, const double* data, int ncenters, double* centers, int* clusters) {
        ncenters = initialize(ndim, nobs, data, ncenters, centers); 
        return hw.run(ndim, nobs, data, ncenters, centers, clusters);
    }

public:
    /**
     * @brief Full statistics from k-means clustering.
     */
    struct Results {
        /**
         * @cond
         */
        Results(int ndim, int nobs, int ncenters) : centers(ndim * ncenters), clusters(nobs) {}
        /**
         * @endcond
         */

        /**
         * A column-major `ndim`-by-`ncenters` array containing per-cluster centroid coordinates.
         */
        std::vector<double> centers;

        /**
         * An array of length `nobs` containing 0-indexed cluster assignments for each observation.
         */
        std::vector<int> clusters;

        /**
         * Further details from the Hartigan-Wong algorithm.
         */
        HartiganWong::Details details;
    };

    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     *
     * @return `centers` and `clusters` are filled, and a `Results` object is returned containing clustering statistics.
     * See `run()` for more details.
     */
    Results run(int ndim, int nobs, const double* data, int ncenters) {
        Results output(ndim, nobs, ncenters);
        ncenters = initialize(ndim, nobs, data, ncenters, output.centers.data()); 
        output.details = hw.run(ndim, nobs, data, ncenters, output.centers.data(), output.clusters.data());
        return output;
    }
};

}

#endif
