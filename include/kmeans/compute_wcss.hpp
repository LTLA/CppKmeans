#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <algorithm>
#include "SimpleMatrix.hpp"

/**
 * @file compute_wcss.hpp
 * @brief Compute within-cluster sum of squares.
 */

namespace kmeans {

/**
 * @tparam Matrix_ Matrix type for the input data, satisfying the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centers and output.
 *
 * @param[in] data A matrix-like object (see `MockMatrix`) containing per-observation data.
 * @param ncenters Number of cluster centers.
 * @param[in] centers Pointer to an array where columns are cluster centers and rows are dimensions (from `data.num_dimensions()`).
 * This should contain the initial centroid locations for each cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * This should contain the 0-based cluster assignment for each observation.
 * @param[out] wcss Pointer to an array of length equal to the number of cluster centers.
 * On output, this will contain the within-cluster sum of squares.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
void compute_wcss(const Matrix_& data, Cluster_ ncenters, const Float_* centers, const Cluster_* clusters, Float_* wcss) {
    auto nobs = data.num_observations();
    auto ndim = data.num_dimensions();
    size_t long_ndim = ndim;
    std::fill_n(wcss, ncenters, 0);

    auto work = data.create_workspace(static_cast<decltype(nobs)>(0), nobs);
    for (decltype(nobs) obs = 0; obs < nobs; ++obs) {
        auto cen = clusters[obs];
        auto curcenter = centers + static_cast<size_t>(cen) * long_ndim;
        auto& curwcss = wcss[cen];

        auto curdata = data.get_observation(work);
        for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++curcenter, ++curdata) {
            auto delta = *curdata - *curcenter;
            curwcss += delta * delta;
        }
    }
}

}

#endif
