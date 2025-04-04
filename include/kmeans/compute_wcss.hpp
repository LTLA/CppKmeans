#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <algorithm>
#include "Matrix.hpp"

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
 * @param data A matrix-like object (see `MockMatrix`) containing per-observation data.
 * @param ncenters Number of cluster centers.
 * @param[in] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * Each column should contain the initial centroid location for its cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * This should contain the 0-based cluster assignment for each observation.
 * @param[out] wcss Pointer to an array of length equal to the number of cluster centers.
 * On output, this will contain the within-cluster sum of squares.
 */
template<class Matrix_, typename Cluster_, typename Float_>
void compute_wcss(const Matrix_& data, Cluster_ ncenters, const Float_* centers, const Cluster_* clusters, Float_* wcss) {
    auto nobs = data.num_observations();
    size_t ndim = data.num_dimensions();
    std::fill_n(wcss, ncenters, 0);

    auto work = data.new_extractor(static_cast<Index<Matrix_> >(0), nobs);
    for (Index<Matrix_> obs = 0; obs < nobs; ++obs) {
        auto curdata = work->get_observation();
        auto cen = clusters[obs];
        auto curcenter = centers + static_cast<size_t>(cen) * ndim; // cast to size_t to avoid overflow.

        Float_& curwcss = wcss[cen];
        for (size_t d = 0; d < ndim; ++d) {
            Float_ delta = static_cast<Float_>(curdata[d]) - curcenter[d]; // cast for consistent precision regardless of Data_.
            curwcss += delta * delta;
        }
    }
}

}

#endif
