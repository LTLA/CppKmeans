#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <vector>

namespace kmeans {

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int>
std::vector<DATA_t> compute_wcss(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, const DATA_t* centers, const CLUSTER_t* clusters) {
    std::vector<DATA_t> wcss(ncenters);
    for (INDEX_t obs = 0; obs < nobs; ++obs) {
        auto cen = clusters[obs];
        auto curcenter = centers + cen * ndim;
        auto& curwcss = wcss[cen];

        auto curdata = data + obs * ndim;
        for (int dim = 0; dim < ndim; ++dim, ++curcenter, ++curdata) {
            curwcss += (*curdata - *curcenter) * (*curdata - *curcenter);
        }
    }

    return wcss;
}

}

#endif
