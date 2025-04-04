#ifndef KMEANS_IS_EDGE_CASE_HPP
#define KMEANS_IS_EDGE_CASE_HPP

#include <numeric>
#include <algorithm>

#include "Details.hpp"
#include "Matrix.hpp"
#include "compute_centroids.hpp"

namespace kmeans {

namespace internal {

template<typename Index_, typename Cluster_>
bool is_edge_case(Index_ nobs, Cluster_ ncenters) {
    return (ncenters <= 1 || static_cast<Index_>(ncenters) >= nobs);
}

template<class Matrix_, typename Cluster_, typename Float_>
Details<Index<Matrix_> > process_edge_case(const Matrix_& data, Cluster_ ncenters, Float_* centers, Cluster_* clusters) {
    auto nobs = data.num_observations();

    if (ncenters == 1) {
        // All points in cluster 0.
        std::fill_n(clusters, nobs, 0);
        std::vector<Index<Matrix_> > sizes(1, nobs);
        compute_centroid(data, centers);
        return Details(std::move(sizes), 0, 0);

    } else if (static_cast<Index<Matrix_> >(ncenters) >= nobs) {
        // Special case, each observation is a center.
        std::iota(clusters, clusters + nobs, 0);
        std::vector<Index<Matrix_> > sizes(ncenters);
        std::fill_n(sizes.begin(), nobs, 1);

        auto ndim = data.num_dimensions();
        auto work = data.new_extractor(static_cast<Index<Matrix_> >(0), nobs);
        auto cptr = centers;
        for (decltype(nobs) o = 0; o < nobs; ++o, cptr += ndim) {
            auto ptr = work->get_observation();
            std::copy_n(ptr, ndim, cptr);
        }

        return Details(std::move(sizes), 0, 0);

    } else { //i.e., ncenters == 0, provided is_edge_case is true.
        return Details<Index<Matrix_> >(0, 0);
    }
}

}

}

#endif
