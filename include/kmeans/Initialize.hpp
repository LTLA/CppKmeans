#ifndef KMEANS_INITIALIZE_HPP
#define KMEANS_INITIALIZE_HPP

#include "SimpleMatrix.hpp"

/**
 * @file Initialize.hpp
 * @brief Interface for k-means initialization.
 */

namespace kmeans {

/**
 * @brief Base class for initialization algorithms.
 *
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class Initialize {
public:
    /**
     * @cond
     */
    virtual ~Initialize() = default;
    /**
     * @endcond
     */

    /**
     * @param data A matrix-like object (see `MockMatrix`) containing per-observation data.
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
