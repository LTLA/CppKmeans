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
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam INDEX_t Integer type for the observation index.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class InitializeNone : public Initialize<DATA_t, CLUSTER_t, INDEX_t> { 
public:
    /*
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param data Pointer to an array where the dimensions are rows and the observations are columns.
     * Data should be stored in column-major format.
     * @param ncenters Number of centers to pick.
     * @param centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * This is left unchanged.
     * @param clusters Ignored in this method.
     *
     * @return The smaller of `ncenters` and `nobs` is returned, see `Initialize::run()`.
     */
    CLUSTER_t run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        return std::min(nobs, static_cast<INDEX_t>(ncenters));
    }
};

}

#endif
