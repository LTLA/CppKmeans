#ifndef KMEANS_REFINE_HPP
#define KMEANS_REFINE_HPP

#include "Details.hpp"
#include "SimpleMatrix.hpp"

/**
 * @file Refine.hpp
 * @brief Interface for k-means refinement.
 */

namespace kmeans {

/**
 * @brief Interface for all k-means refinement algorithms.
 *
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class Refine {
public:
    virtual ~Refine() = default;

    /**
     * @param data A matrix-like object (see `MockMatrix`) containing per-observation data.
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
    virtual Details<typename Matrix_::index_type> run(const Matrix_& data, Cluster_ num_centers, Float_* centers, Cluster_* clusters) const = 0;
};

}

#endif
