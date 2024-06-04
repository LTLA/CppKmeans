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
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Center_ Floating-point type for the centroids.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
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
     * @param[in] data A matrix-like object (see `SimpleMatrix`) containing per-observation data.
     * @param ncenters Number of cluster centers.
     * @param[out] centers Pointer to an array of length equal to `ncenters` multiplied by the number of dimensions in `data`.
     * On output, this will contain the final centroid locations for each cluster, 
     * where columns are cluster centers and rows are dimensions in column-major order.
     *
     * @return `centers` is filled with the new cluster centers.
     * The number of filled centers is returned - this is usually equal to `ncenters`, but may not be if, e.g., `ncenters` is greater than the number of observations.
     * If the returned value is less than `ncenters`, only the first few centers in `centers` will be filled.
     */
    virtual Cluster_ run(const Matrix_& data, Cluster_ ncenters, Center_* centers) const = 0;
};

}

#endif
