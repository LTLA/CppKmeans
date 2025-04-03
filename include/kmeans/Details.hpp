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
template<typename Index_>
struct Details {
    /**
     * @cond
     */
    Details() = default;

    Details(int iterations, int status) : sizes(0), iterations(iterations), status(status) {}

    Details(std::vector<Index_> sizes, int iterations, int status) : sizes(std::move(sizes)), iterations(iterations), status(status) {} 
    /**
     * @endcond
     */

    /**
     * The number of observations in each cluster.
     * Some clusters may be empty, e.g., when there are more requested centers than clusters.
     */
    std::vector<Index_> sizes;

    /**
     * The number of iterations used to achieve convergence.
     * This value may be greater than the `maxit` if convergence was not achieved, see `status`.
     */
    int iterations = 0;

    /**
     * The status of the algorithm on completion.
     * A value of 0 indicates that the algorithm completed successfully.
     * The interpretation of a non-zero value depends on the algorithm.
     */
    int status = 0;
};

}

#endif

