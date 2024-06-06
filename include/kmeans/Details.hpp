#ifndef KMEANS_DETAILS_HPP
#define KMEANS_DETAILS_HPP

#include <vector>

/**
 * @file Details.hpp
 *
 * @brief Report detailed clustering statistics.
 */

namespace kmeans {

/**
 * @brief Additional statistics from the k-means algorithm.
 *
 * @tparam Index_ Integer type for the observation index.
 */
template<typename Index_ = int>
struct Details {
    /**
     * @cond
     */
    Details() : iterations(0), status(0) {}

    Details(int it, int st) : sizes(0), iterations(it), status(st) {}

    Details(std::vector<Index_> s, int it, int st) : sizes(std::move(s)), iterations(it), status(st) {} 
    /**
     * @endcond
     */

    /**
     * The number of observations in each cluster.
     * All values are guaranteed to be positive for non-zero numbers of observations when `status == 0`.
     */
    std::vector<Index_> sizes;

    /**
     * The number of iterations used to achieve convergence.
     * This value may be greater than the `maxit` if convergence was not achieved, see `status`.
     */
    int iterations;

    /**
     * The status of the algorithm on completion.
     * A value of 0 indicates that the algorithm completed successfully.
     * The interpretation of a non-zero value depends on the algorithm.
     */
    int status;
};

}

#endif

