#ifndef KMEANS_INITIALIZATION_HPP
#define KMEANS_INITIALIZATION_HPP 

#include <vector>
#include "random.hpp"
#include "aarand/aarand.hpp"
#include <iostream>

/**
 * @file initialization.hpp
 *
 * @brief Initialization methods to choose the starting centroids.
 */

namespace kmeans {

/**
 * @cond
 */
template<typename DATA_t, typename INDEX_t, class ENGINE>
INDEX_t weighted_sample(const std::vector<DATA_t>& cumulative, const std::vector<DATA_t>& mindist, INDEX_t nobs, ENGINE& eng) {
    auto total = cumulative.back();
    INDEX_t chosen_id = 0;

    do {
        const DATA_t sampled_weight = total * aarand::standard_uniform(eng);
        chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();

        if (chosen_id == nobs) {
            std::cout << chosen_id << "\t" << "YAY" << std::endl;
        } else if (mindist[chosen_id] == 0) {
            std::cout << chosen_id << "\t" << "YAY2" << std::endl;
        }

        // We wrap this in a do/while to defend against edge cases where
        // ties are chosen. The most obvious of these is when you get a
        // `sampled_weight` of zero _and_ there exists a bunch of zeros at
        // the start of `cumulative`. One could also get unexpected ties
        // from limited precision in floating point comparisons, so we'll
        // just be safe and implement a loop here, in the same vein as
        // uniform01.
    } while (chosen_id == nobs || mindist[chosen_id] == 0);

    return chosen_id;
}
/**
 * @endcond
 */

/**
 * Implements the <b>k-means++</b> initialization described by Arthur and Vassilvitskii (2007).
 * This approach involves the selection of starting points via iterations of weighted sampling, 
 * where the sampling probability for each point is proportional to the squared distance to the closest starting point that was chosen in any of the previous iterations.
 * The aim is to obtain well-separated starting points to encourage the formation of suitable clusters.
 *
 * @param ndim Number of dimensions.
 * @param nobs Number of observations.
 * @param data Pointer to an array where the dimensions are rows and the observations are columns.
 * Data should be stored in column-major format.
 * @param ncenters Number of centers to pick.
 * @param eng An instance of a random number engine.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam INDEX_t Integer type for the observation index.
 * This should be at least 50 times greater than the maximum expected number of observations.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam ENGINE A random number engine, e.g., `std::mt19937`.
 *
 * @return A vector of indices for the observations that were selected as starting points.
 * Note that the length may be less than `ncenters` if `ncenters > no` (in which case each observation is chosen as a starting point)
 * or if there are duplicate observations (in which case only one of each set of duplicates is chosen).
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
std::vector<INDEX_t> weighted_initialization(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, ENGINE& eng) {
    std::vector<DATA_t> mindist(nobs, 1);
    std::vector<DATA_t> cumulative(nobs);
    std::vector<INDEX_t> sofar;

    if (!nobs) {
        return sofar;
    }
    sofar.reserve(ncenters);

    for (CLUSTER_t cen = 0; cen < ncenters; ++cen) {
        INDEX_t counter = 0;
        if (!sofar.empty()) {
            auto last = sofar.back();

            #pragma omp parallel for
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                if (mindist[obs]) {
                    const DATA_t* acopy = data + obs * ndim;
                    const DATA_t* scopy = data + last * ndim;
                    DATA_t r2 = 0;
                    for (int dim = 0; dim < ndim; ++dim, ++acopy, ++scopy) {
                        r2 += (*acopy - *scopy) * (*acopy - *scopy);
                    }

                    if (cen == 1 || r2 < mindist[obs]) {
                        mindist[obs] = r2;
                    }
                }
            }
        } else {
            counter = nobs;
        }

        cumulative[0] = mindist[0];
        for (INDEX_t i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }

        const auto total = cumulative.back();
        if (total == 0) { // a.k.a. only duplicates left.
            break;
        }

        auto chosen_id = weighted_sample(cumulative, mindist, nobs, eng);
        mindist[chosen_id] = 0;
        sofar.push_back(chosen_id);
    }

    return sofar;
}

/**
 * Implements a simple initialization of the starting points where random observations are sampled without replacement.
 *
 * @tparam INDEX_t Integer type for the observation index.
 * This should be at least 50 times greater than the maximum expected number of observations.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam ENGINE A random number engine, e.g., `std::mt19937`.
 *
 * @param nobs Number of observations.
 * @param ncenters Number of centers for which to pick starting points.
 * @param eng An instance of a random number engine.
 * 
 * @return A vector of indices for the observations that were selected as starting points.
 * Note that the length may be less than `ncenters` if `ncenters > no`, in which case each observation is chosen as a starting point.
 */
template<typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
std::vector<INDEX_t> simple_initialization(INDEX_t nobs, CLUSTER_t ncenters, ENGINE& eng) {
    return sample_without_replacement(nobs, ncenters, eng);
}

}

#endif
