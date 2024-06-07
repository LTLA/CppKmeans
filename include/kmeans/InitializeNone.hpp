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
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * 
 * This class assumes that that cluster centers are already present in the `centers` array, and returns them without modification.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class InitializeNone : public Initialize<Matrix_, Cluster_, Float_> { 
public:
    Cluster_ run(const Matrix_& matrix, Cluster_ ncenters, Float_*) const {
        return std::min(matrix.num_observations(), static_cast<typename Matrix_::index_type>(ncenters));
    }
};

}

#endif
