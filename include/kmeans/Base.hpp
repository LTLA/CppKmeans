#ifndef KMEANS_BASE_HPP
#define KMEANS_BASE_HPP

#include "Details.hpp"

namespace kmeans {

/**
 * @brief Base class for all k-means refinement algorithms.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class Refine {
public:
    virtual ~Refine() {}

    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On input, this should contain the initial centroid locations for each cluster.
     * Data should be stored in column-major order.
     * On output, this will contain the final centroid locations for each cluster.
     * @param[out] clusters Pointer to an array of length `nobs`.
     * On output, this will contain the cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * If `ncenters > nobs`, only the first `nobs` columns of the `centers` array will be filled.
     */
    virtual Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) = 0;
};

/**
 * @brief Base class for all k-means initialization algorithms.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class Initialize {
public:
    virtual ~Initialize() {}

    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On input, this should contain the initial centroid locations for each cluster, or it may be ignored, depending on the subclass.
     * Data should be stored in column-major order.
     * On output, this will contain the final centroid locations for each cluster.
     * @param[in, out] clusters Pointer to an array of length `nobs`.
     * On input, this should contain the initial cluster assignments for each observation, or it may be ignored, depending on the subclass.
     * This may be modified by the subclass - no guarantees are provided as to the output state.
     *
     * @return `centers` is filled with the new cluster centers.
     * The number of filled centers is returned - this is usually equal to `ncenters`, but may not be if, e.g., `ncenters > nobs`.
     * If the returned value is less than `ncenters`, only the first centers in `centers` will be filled.
     */
    virtual CLUSTER_t run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) = 0;
};


}

#endif
