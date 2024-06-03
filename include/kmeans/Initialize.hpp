#ifndef KMEANS_BASE_HPP
#define KMEANS_BASE_HPP

#include "Details.hpp"

namespace kmeans {

/**
 * @brief Base class for initialization algorithms.
 *
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Index_ Integer type for the observation index.
 */
template<typename Data_ = double, typename Cluster_ = int, typename Index_ = int>
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
    virtual Cluster_ run(const Matrix_& data, Cluster_ ncenters, Data_* centers) const = 0;
};

}

#endif
