#ifndef KMEANS_COMPUTE_CENTROIDS_HPP
#define KMEANS_COMPUTE_CENTROIDS_HPP

#include <algorithm>

namespace kmeans {

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class V>
void compute_centroids(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, const CLUSTER_t* clusters, const V& sizes) {
    std::fill(centers, centers + ndim * ncenters, 0);

    for (INDEX_t obs = 0; obs < nobs; ++obs) {
        auto copy = centers + clusters[obs] * ndim;
        auto mine = data + obs * ndim;
        for (int dim = 0; dim < ndim; ++dim, ++copy, ++mine) {
            *copy += *mine;
        }
    }

    for (CLUSTER_t cen = 0; cen < ncenters; ++cen) {
        if (sizes[cen]) {
            auto curcenter = centers + cen * ndim;
            for (int dim = 0; dim < ndim; ++dim, ++curcenter) {
                *curcenter /= sizes[cen];
            }
        }
    }
}

}

#endif
