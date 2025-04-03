#ifndef KMEANS_REFINE_HPP
#define KMEANS_REFINE_HPP

#include "Details.hpp"
#include "Matrix.hpp"

/**
 * @file Refine.hpp
 * @brief Interface for k-means refinement.
 */

namespace kmeans {

/**
 * @brief Interface for k-means refinement algorithms.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the data.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Type for the input data matrix.
 * This should satisfy the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, typename Matrix_ = Matrix<Index_, Data_> >
class Refine {
public:
    /**
     * @cond
     */
    Refine() = default;
    Refine(Refine&&) = default;
    Refine(const Refine&) = default;
    Refine& operator=(Refine&&) = default;
    Refine& operator=(const Refine&) = default;
    virtual ~Refine() = default;

    static_assert(std::is_same<decltype(std::declval<Matrix_>().num_observations()), Index_>::value);
    static_assert(std::is_same<typename std::remove_pointer<decltype(std::declval<Matrix_>().new_extractor()->get_observation(0))>::type, const Data_>::value);
    /**
     * @endcond
     */

    /**
     * @param data A matrix-like object containing per-observation data.
     * @param num_centers Number of cluster centers.
     * @param[in, out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
     * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
     * On input, each column should contain the initial centroid location for its cluster.
     * On output, each column will contain the final centroid locations for each cluster.
     * @param[out] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
     * On output, this will contain the cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * If `num_centers` is greater than `data.num_observations()`, only the first `data.num_observations()` columns of the `centers` array will be filled.
     */
    virtual Details<Index_> run(const Matrix_& data, Cluster_ num_centers, Float_* centers, Cluster_* clusters) const = 0;
};

}

#endif
