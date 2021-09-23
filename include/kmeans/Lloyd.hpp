#ifndef KMEANS_LLOYD_HPP
#define KMEANS_LLOYD_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <limits>

#include "Base.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "is_edge_case.hpp"
#include "compute_centroids.hpp"
#include "compute_wcss.hpp"

/**
 * @file Lloyd.hpp
 *
 * @brief Implements the Lloyd algorithm for k-means clustering.
 */

namespace kmeans {

/**
 * @brief Implements the Lloyd algorithm for k-means clustering.
 *
 * The Lloyd algorithm is the simplest k-means clustering algorithm,
 * involving several iterations of batch assignments and center calculations.
 * Specifically, we assign each observation to its closest cluster, and once all points are assigned, we recompute the cluster centroids.
 * This is repeated until there are no reassignments or the maximum number of iterations is reached.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 *
 * @see
 * Lloyd, S. P. (1982).  
 * Least squares quantization in PCM.
 * _IEEE Transactions on Information Theory_ 28, 128-137.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class Lloyd : public Base<DATA_t, CLUSTER_t, INDEX_t> {
public:
    /** 
     * @brief Default parameter values for `Lloyd`.
     */
    struct Defaults {
        /** 
         * See `Lloyd::set_max_iterations()`.
         */
        static constexpr int max_iterations = 10;
    };

private:
    int maxiter = Defaults::max_iterations;

public:
    /**
     * @param m Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     *
     * @return A reference to this `Lloyd` object.
     */
    Lloyd& set_max_iterations(int m = Defaults::max_iterations) {
        maxiter = m;
        return *this;
    }

public:
    Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        if (is_edge_case(nobs, ncenters)) {
            return process_edge_case(ndim, nobs, data, ncenters, centers, clusters);
        }

        int iter = 0, status = 0;
        std::vector<INDEX_t> sizes(ncenters);
        std::vector<CLUSTER_t> copy(nobs);

        for (iter = 1; iter <= maxiter; ++iter) {
            // Nearest-neighbor search to assign to the closest cluster.
            // Note that we move the `updated` check outside of this loop
            // so that, in the future, this is more easily parallelized.
            QuickSearch<DATA_t, CLUSTER_t> index(ndim, ncenters, centers);
            #pragma omp parallel for
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                copy[obs] = index.find(data + obs * ndim);
            }

            bool updated = false;
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                if (copy[obs] != clusters[obs]) {
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                break;
            }
            std::copy(copy.begin(), copy.end(), clusters);

            // Counting the number in each cluster.
            std::fill(sizes.begin(), sizes.end(), 0);
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                ++sizes[clusters[obs]];
            }

            for (CLUSTER_t c = 0; c < ncenters; ++ c) {
                if (!sizes[c]) {
                    status = 1;
                    break;
                }
            }
            
            compute_centroids(ndim, nobs, data, ncenters, centers, clusters, sizes);
        }

        if (iter == maxiter + 1) {
            status = 2;
        }

        return Details<DATA_t, INDEX_t>(
            std::move(sizes),
            compute_wcss(ndim, nobs, data, ncenters, centers, clusters),
            iter, 
            status
        );
    }
};

}

#endif
