#ifndef KMEANS_REFINE_HPP
#define KMEANS_REFINE_HPP

#include "Details.hpp"
#include "SimpleMatrix.hpp"

namespace kmeans {

/**
 * @brief Interface for all k-means refinement algorithms.
 *
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Center_ Floating-point type for the centroids.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
class Refine {
public:
    virtual ~Refine() = default;

    /**
     * @param[in] data A matrix-like object (see `MockMatrix`) containing per-observation data.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to an array where columns are cluster centers and rows are dimensions (from `data.num_dimensions()`).
     * On input, this should contain the initial centroid locations for each cluster.
     * Data should be stored in column-major order.
     * On output, this will contain the final centroid locations for each cluster.
     * @param[out] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
     * On output, this will contain the cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled, and a `Details` object is returned containing clustering statistics.
     * If `ncenters > nobs`, only the first `nobs` columns of the `centers` array will be filled.
     */
    virtual Details<typename Matrix_::index_type> run(const Matrix_& data, Cluster_ ncenters, Center_* centers, Cluster_* clusters) const = 0;
};

}

#endif
