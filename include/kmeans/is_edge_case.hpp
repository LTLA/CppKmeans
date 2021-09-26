#ifndef KMEANS_IS_EDGE_CASE_HPP
#define KMEANS_IS_EDGE_CASE_HPP

#include <numeric>
#include <algorithm>
#include "compute_wcss.hpp"
#include "compute_centroids.hpp"

namespace kmeans {

template<typename INDEX_t = int, typename CLUSTER_t = int>
bool is_edge_case(INDEX_t nobs, CLUSTER_t ncenters) {
    return (ncenters <= 1 || static_cast<INDEX_t>(ncenters) >= nobs);
}

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int>
Details<DATA_t, INDEX_t> process_edge_case(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
    if (ncenters == 1) {
        // All points in cluster 0.
        std::fill(clusters, clusters + nobs, 0);
        std::vector<INDEX_t> sizes(1, nobs);
        compute_centroids(ndim, nobs, data, ncenters, centers, clusters, sizes);
        auto wcss = compute_wcss(ndim, nobs, data, ncenters, centers, clusters);
        return Details(std::move(sizes), std::move(wcss), 0, 0);

    } else if (ncenters >= nobs) {
        // Special case, each observation is a center.
        std::iota(clusters, clusters + nobs, 0);
        std::vector<INDEX_t> sizes(ncenters);
        std::fill(sizes.begin(), sizes.begin() + nobs, 1);
        compute_centroids(ndim, nobs, data, ncenters, centers, clusters, sizes);
        auto wcss = compute_wcss(ndim, nobs, data, ncenters, centers, clusters);
        return Details(std::move(sizes), std::move(wcss), 0, (ncenters > nobs ? 3 : 0));

    } else { //i.e., ncenters == 0, provided is_edge_case is true.
        return Details<DATA_t, INDEX_t>(0, 3);
    }
}

}

#endif
