#ifndef KMEANS_COMPUTE_CENTROIDS_HPP
#define KMEANS_COMPUTE_CENTROIDS_HPP

#include <algorithm>

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void compute_centroid(const Matrix_& data, Float_* center) {
    auto ndim = data.num_dimensions();
    std::fill_n(center, ndim, 0);
    auto nobs = data.num_observations();
    auto work = data.create_workspace(static_cast<typename Matrix_::index_type>(0), nobs);

    for (decltype(nobs) i = 0; i < nobs; ++i) {
        auto dptr = data.get_observation(work);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            center[d] += static_cast<Float_>(dptr[d]); // cast for consistent precision regardless of Matrix_::data_type.
        }
    }

    for (decltype(ndim) d = 0; d < ndim; ++d) {
        center[d] /= nobs;
    }
}

template<class Matrix_, typename Cluster_, typename Float_>
void compute_centroids(const Matrix_& data, Cluster_ ncenters, Float_* centers, const Cluster_* clusters, const std::vector<typename Matrix_::index_type>& sizes) {
    auto nobs = data.num_observations();
    auto ndim = data.num_dimensions();
    size_t long_ndim = ndim;
    std::fill(centers, centers + long_ndim * static_cast<size_t>(ncenters), 0); // cast to avoid overflow.

    auto work = data.create_workspace(static_cast<typename Matrix_::index_type>(0), nobs);
    for (decltype(nobs) obs = 0; obs < nobs; ++obs) {
        auto copy = centers + static_cast<size_t>(clusters[obs]) * long_ndim;
        auto mine = data.get_observation(work);
        for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++copy, ++mine) {
            *copy += static_cast<Float_>(*mine); // cast for consistent precision regardless of Matrix_::data_type.

        }
    }

    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        auto s = sizes[cen];
        if (s) {
            auto curcenter = centers + static_cast<size_t>(cen) * long_ndim; // cast to avoid overflow.
            for (int dim = 0; dim < ndim; ++dim, ++curcenter) {
                *curcenter /= s;
            }
        }
    }
}

}

}

#endif
