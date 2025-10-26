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
 * @tparam Index_ Integer type of the observation index.
 */
template<typename Index_>
struct Details {
    /**
     * @cond
     */
    Details() = default;

    Details(const int iterations, const int status) : sizes(0), iterations(iterations), status(status) {}

    Details(std::vector<Index_> sizes, const int iterations, const int status) : sizes(std::move(sizes)), iterations(iterations), status(status) {} 
    /**
     * @endcond
     */

    /**
     * The number of observations in each cluster.
     * Some clusters may be empty, e.g., when there are more requested centers than clusters -
     * see `remove_unused_centers()` to optionally remove such clusters.
     */
    std::vector<Index_> sizes;

    /**
     * The number of iterations that were performed.
     * This can be interpreted as the number of iterations to convergence if `Details::status == 0`.
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

