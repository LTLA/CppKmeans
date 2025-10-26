#ifndef KMEANS_REFINE_HPP
#define KMEANS_REFINE_HPP

#include <utility>
#include <type_traits>

#include "Details.hpp"
#include "Matrix.hpp"
#include "utils.hpp"

/**
 * @file Refine.hpp
 * @brief Interface for k-means refinement.
 */

namespace kmeans {

/**
 * @brief Interface for k-means refinement algorithms.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, typename Matrix_ = Matrix<Index_, Data_> >
class Refine {
public:
    /**
     * @cond
     */
    Refine() = default;
    Refine(Refine&&) = default;
    Refine(const Refine&) = default;
    Refine& operator=(Refine&&) = default;
    Refine& operator=(const Refine&) = default;
    virtual ~Refine() = default;

    static_assert(std::is_same<I<decltype(std::declval<Matrix_>().num_observations())>, Index_>::value);
    static_assert(std::is_same<I<decltype(*(std::declval<Matrix_>().new_extractor()->get_observation(0)))>, Data_>::value);
    /**
     * @endcond
     */

    /**
     * @param data A matrix containing data for each observation.
     * @param num_centers Number of cluster centers.
     * @param[in, out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
     * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
     * On input, column `j` should contain the initial centroid location for cluster `j`.
     * On output, each column will contain the final centroid location for its corresponding cluster.
     * @param[out] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
     * On output, this will contain the 0-based cluster assignment for each observation.
     * Specifically, each entry is an index that refers to a column of `centers` and is no greater than `num_centers`.
     *
     * @return `centers` and `clusters` are filled, and an object is returned containing clustering statistics.
     *
     * Not all columns of `centers` may be represented in the output `clusters`, i.e., some clusters may be unused.
     * The `remove_unused_centers()` function will rearrange the cluster assignments to more easily skip these empty clusters.
     * In practice, empty clusters should be rare if the initial centroids are chosen appropriately.
     */
    virtual Details<Index_> run(const Matrix_& data, Cluster_ num_centers, Float_* centers, Cluster_* clusters) const = 0;
};

}

#endif
