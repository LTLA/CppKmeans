#ifndef KMEANS_INITIALIZE_HPP
#define KMEANS_INITIALIZE_HPP

#include "Matrix.hpp"

#include <type_traits>

/**
 * @file Initialize.hpp
 * @brief Interface for k-means initialization.
 */

namespace kmeans {

/**
 * @brief Interface for k-means initialization algorithms.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class Initialize {
public:
    /**
     * @cond
     */
    Initialize() = default;
    Initialize(Initialize&&) = default;
    Initialize(const Initialize&) = default;
    Initialize& operator=(Initialize&&) = default;
    Initialize& operator=(const Initialize&) = default;
    virtual ~Initialize() = default;

    static_assert(std::is_same<decltype(std::declval<Matrix_>().num_observations()), Index_>::value);
    static_assert(std::is_same<typename std::remove_pointer<decltype(std::declval<Matrix_>().new_extractor()->get_observation(0))>::type, const Data_>::value);
    /**
     * @endcond
     */

    /**
     * @param data A matrix-like object containing per-observation data.
     * @param num_centers Number of cluster centers.
     * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
     * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
     * On output, each column will contain the final centroid locations for each cluster.
     *
     * @return `centers` is filled with the new cluster centers.
     * The number of filled centers is returned - this is usually equal to `num_centers`, but may not be if, e.g., `num_centers` is greater than the number of observations.
     * If the returned value is less than `num_centers`, only the first few centers in `centers` will be filled.
     */
    virtual Cluster_ run(const Matrix_& data, Cluster_ num_centers, Float_* centers) const = 0;
};

}

#endif
