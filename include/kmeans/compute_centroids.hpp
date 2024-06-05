#ifndef KMEANS_COMPUTE_CENTROIDS_HPP
#define KMEANS_COMPUTE_CENTROIDS_HPP

#include <algorithm>

namespace kmeans {

namespace internal {

template<class Matrix_, typename Cluster_, typename Center_>
void compute_centroids(const Matrix_& data, Cluster_ ncenters, Center_* centers, const Cluster_* clusters, const std::vector<typename Matrix_::index_type>& sizes) {
    auto nobs = data.num_observations();
    auto ndim = data.num_dimensions();
    size_t long_ndim = ndim;
    std::fill(centers, centers + long_ndim * static_cast<size_t>(ncenters), 0); // cast to avoid overflow.

    auto work = data.create_workspace(0, nobs);
    for (decltype(nobs) obs = 0; obs < nobs; ++obs) {
        auto copy = centers + static_cast<size_t>(clusters[obs]) * long_ndim;
        auto mine = data.get_observation(work);
        for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++copy, ++mine) {
            *copy += *mine;
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
