#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <vector>

namespace kmeans {

namespace internal {

template<class Matrix_, typename Cluster_, typename Center_>
void compute_wcss(const Matrix_& data, Cluster_ ncenters, const Center_* centers, const Cluster_* clusters, Center_* wcss) {
    auto nobs = data.num_observations();
    auto ndim = data.num_dimensions();
    std::fill_n(wcss, ncenters, 0);

    auto work = data.create_workspace(0, nobs);
    for (decltype(nobs) obs = 0; obs < nobs; ++obs) {
        auto cen = clusters[obs];
        auto curcenter = centers + static_cast<size_t>(cen) * static_cast<size_t>(ndim);
        auto& curwcss = wcss[cen];

        auto curdata = data.get_observation(work);
        for (int dim = 0; dim < ndim; ++dim, ++curcenter, ++curdata) {
            auto delta = *curdata - *curcenter;
            curwcss += delta * delta;
        }
    }
}

}

}

#endif
