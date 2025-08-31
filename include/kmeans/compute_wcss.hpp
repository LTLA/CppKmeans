#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "Matrix.hpp"
#include "utils.hpp"

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
void compute_wcss(const Matrix_& data, const Cluster_ ncenters, const Float_* const centers, const Cluster_* const clusters, Float_* const wcss) {
    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    std::fill_n(wcss, ncenters, 0);

    auto work = data.new_extractor(static_cast<Index<Matrix_> >(0), nobs);
    for (decltype(I(nobs)) obs = 0; obs < nobs; ++obs) {
        const auto curdata = work->get_observation();
        const auto cen = clusters[obs];

        Float_& curwcss = wcss[cen];
        for (decltype(I(ndim)) d = 0; d < ndim; ++d) {
            const auto curcenter = centers[sanisizer::nd_offset<std::size_t>(d, ndim, cen)];
            const Float_ delta = static_cast<Float_>(curdata[d]) - curcenter; // cast for consistent precision regardless of the input data type.
            curwcss += delta * delta;
        }
    }
}

}

#endif
