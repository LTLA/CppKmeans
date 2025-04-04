#ifndef KMEANS_COMPUTE_CENTROIDS_HPP
#define KMEANS_COMPUTE_CENTROIDS_HPP

#include <algorithm>
#include "Matrix.hpp"

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void compute_centroid(const Matrix_& data, Float_* center) {
    size_t ndim = data.num_dimensions();
    std::fill_n(center, ndim, 0);
    auto nobs = data.num_observations();

    auto work = data.new_extractor(static_cast<Index<Matrix_> >(0), nobs);
    for (Index<Matrix_> i = 0; i < nobs; ++i) {
        auto dptr = work->get_observation();
        for (size_t d = 0; d < ndim; ++d) {
            center[d] += static_cast<Float_>(dptr[d]); // cast for consistent precision regardless of the matrix datatype.
        }
    }

    for (size_t d = 0; d < ndim; ++d) {
        center[d] /= nobs;
    }
}

template<class Matrix_, typename Cluster_, typename Float_>
void compute_centroids(const Matrix_& data, Cluster_ ncenters, Float_* centers, const Cluster_* clusters, const std::vector<Index<Matrix_> >& sizes) {
    auto nobs = data.num_observations();
    size_t ndim = data.num_dimensions();
    std::fill(centers, centers + ndim * static_cast<size_t>(ncenters), 0); // cast to size_t to avoid overflow.

    auto work = data.new_extractor(static_cast<Index<Matrix_> >(0), nobs);
    for (Index<Matrix_> obs = 0; obs < nobs; ++obs) {
        auto copy = centers + static_cast<size_t>(clusters[obs]) * ndim;
        auto mine = work->get_observation();
        for (size_t d = 0; d < ndim; ++d) {
            copy[d] += static_cast<Float_>(mine[d]); // cast for consistent precision regardless of the matrix datatype.
        }
    }

    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        auto s = sizes[cen];
        if (s) {
            auto curcenter = centers + static_cast<size_t>(cen) * ndim; // cast to size_t avoid overflow.
            for (size_t d = 0; d < ndim; ++d) {
                curcenter[d] /= s;
            }
        }
    }
}

}

}

#endif
