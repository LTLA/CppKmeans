#ifndef KMEANS_DETAILS_HPP
#define KMEANS_DETAILS_HPP

#include <vector>

namespace kmeans {

/**
 * @file Details.hpp
 *
 * @brief Report detailed clustering statistics.
 */

/**
 * @brief Additional statistics from the k-means algorithm.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam INDEX_t Integer type for the observation index.
 * This should have a maximum positive value that is at least 50 times greater than the maximum expected number of observations.
 */
template<typename DATA_t = double, typename INDEX_t = int>
struct Details {
    /**
     * @cond
     */
    Details() : iterations(0), status(0) {}

    Details(int it, int st) : sizes(0), withinss(0), iterations(it), status(st) {}

    Details(std::vector<INDEX_t> s, std::vector<DATA_t> w, int it, int st) : sizes(std::move(s)), withinss(std::move(w)), iterations(it), status(st) {} 
    /**
     * @endcond
     */

    /**
     * The number of observations in each cluster.
     * All values are guaranteed to be positive for non-zero numbers of observations when `status == 0`.
     */
    std::vector<INDEX_t> sizes;

    /**
     * The within-cluster sum of squares for each cluster.
     * All values are guaranteed to be non-negative.
     * Values may be zero for clusters with only one observation.
     */
    std::vector<DATA_t> withinss;

    /**
     * The number of iterations used to achieve convergence.
     * This value may be greater than the `maxit` if convergence was not achieved, see `status`.
     */
    int iterations;

    /**
     * The status of the algorithm.
     * A value of 0 indicates that the algorithm completed successfully.
     * The interpretation of a non-zero value depends on the algorithm.
     *
     * For `HartiganWong`:
     *
     * - 1: empty cluster detected.
     * This usually indicates a problem with the initial choice of centroids.
     * It can also occur in pathological situations with duplicate points.
     * - 2: maximum iterations reached without convergence. 
     * - 3: the number of centers is not positive or greater than the number of observations.
     * - 4: maximum number of quick transfer steps exceeded.
     *
     * For `Lloyd`:
     *
     * - 1: empty cluster detected.
     * - 2: maximum iterations reached without convergence. 
     */
    int status;
};

}

#endif

