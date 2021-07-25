#ifndef KMEANS_IS_EDGE_CASE_HPP
#define KMEANS_IS_EDGE_CASE_HPP

#include <numeric>
#include <algorithm>

namespace kmeans {
    
template<typename INDEX_t = int, typename CLUSTER_t = int, class V>
bool is_edge_case(INDEX_t nobs, CLUSTER_t ncenters, CLUSTER_t* clusters, V& sizes, int& ifault) {
    if (ncenters == 1) {
        // All points in cluster 0.
        std::fill(clusters, clusters + nobs, 0);
        sizes[0] = nobs;
        return true;

    } else if (ncenters >= nobs) {
        // Special case, each observation is a center.
        std::iota(clusters, clusters + nobs, 0);            
        std::fill(sizes.begin(), sizes.begin() + nobs, 1);
        if (ncenters > nobs) {
            std::fill(sizes.begin() + nobs, sizes.end(), 0);
            ifault = 3;
        }
        return true;

    } else if (ncenters == 0) {
        // No need to fill 'sizes', it's already all-zero on input.
        ifault = 3;
        std::fill(clusters, clusters + nobs, 0);
        return true;
    }

    return false;
}

}

#endif
