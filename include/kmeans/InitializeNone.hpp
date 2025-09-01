#ifndef KMEANS_INITIALIZE_NONE_HPP
#define KMEANS_INITIALIZE_NONE_HPP 

#include "Initialize.hpp"
#include <algorithm>

/**
 * @file InitializeNone.hpp
 *
 * @brief Class for no initialization.
 */

namespace kmeans {

/**
 * @brief No-op "initialization" with existing cluster centers.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 * 
 * This class assumes that that cluster centers are already present in the `centers` array, and returns them without modification.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class InitializeNone final : public Initialize<Index_, Data_, Cluster_, Float_, Matrix_> { 
public:
    /**
     * @cond
     */
    Cluster_ run(const Matrix_& matrix, const Cluster_ ncenters, Float_* const) const {
        return std::min(matrix.num_observations(), static_cast<Index_>(ncenters));
    }
    /**
     * @endcond
     */
};

}

#endif
