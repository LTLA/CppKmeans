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
 * @brief Perform "initialization" by just using the input cluster centers.
 *
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam Index_ Integer type for the observation index.
 */
template<typename Data_ = double, typename CLUSTER_t = int, typename Index_ = int>
class InitializeNone : public Initialize<Data_, CLUSTER_t, Index_> { 
public:
    CLUSTER_t run(int ndim, Index_ nobs, const Data_* data, CLUSTER_t ncenters, Data_* centers, CLUSTER_t* clusters) {
        return std::min(nobs, static_cast<Index_>(ncenters));
    }
};

}

#endif
