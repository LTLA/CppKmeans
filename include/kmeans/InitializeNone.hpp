#ifndef KMEANS_INITIALIZE_NONE_HPP
#define KMEANS_INITIALIZE_NONE_HPP 

#include "Base.hpp"
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
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Center_ Floating-point type for the centroids.
 * 
 * This class assumes that that cluster centers are already present in the `centers` array, and returns them without modification.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
class InitializeNone : public Initialize<Data_, Cluster_, Center_> { 
public:
    Cluster_ run(const Matrix_& matrix, Cluster_ ncenters, Center_*) {
        return std::min(matrix.num_observations(), static_cast<typename Matrix_::index_type>(ncenters));
    }
};

}

#endif
