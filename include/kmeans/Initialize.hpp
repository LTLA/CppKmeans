#ifndef KMEANS_INITIALIZE_HPP
#define KMEANS_INITIALIZE_HPP

#include <utility>
#include <type_traits>

#include "Matrix.hpp"
#include "utils.hpp"

/**
 * @file Initialize.hpp
 * @brief Interface for k-means initialization.
 */

namespace kmeans {

/**
 * @brief Interface for k-means initialization algorithms.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
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

    static_assert(std::is_same<I<decltype(std::declval<Matrix_>().num_observations())>, Index_>::value);
    static_assert(std::is_same<I<decltype(*(std::declval<Matrix_>().new_extractor()->get_observation(0)))>, Data_>::value);
    /**
     * @endcond
     */

    /**
     * @param data A matrix containing data for each observation.
     * @param num_centers Number of cluster centers.
     * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
     * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
     * On output, each column will contain the final centroid locations for each cluster.
     *
     * @return `centers` is filled with the new cluster centers and the number of filled centers is returned.
     * The latter is usually equal to `num_centers`, but may not be if, e.g., `num_centers` is greater than the number of observations.
     * If the returned value is less than `num_centers`, only the left-most columns in `centers` will be filled.
     */
    virtual Cluster_ run(const Matrix_& data, Cluster_ num_centers, Float_* centers) const = 0;
};

}

#endif
