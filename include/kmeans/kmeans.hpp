#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include <vector>

#include "sanisizer/sanisizer.hpp"

#include "Details.hpp"
#include "Refine.hpp"
#include "Initialize.hpp"
#include "Matrix.hpp"
#include "SimpleMatrix.hpp"

#include "InitializeKmeanspp.hpp"
#include "InitializeRandom.hpp"
#include "InitializeVariancePartition.hpp"
#include "InitializeNone.hpp"

#include "RefineHartiganWong.hpp"
#include "RefineLloyd.hpp"
#include "RefineMiniBatch.hpp"

#include "compute_wcss.hpp"

/** 
 * @file kmeans.hpp
 * @brief Perform k-means clustering.
 */

/**
 * @namespace kmeans
 * @brief Perform k-means clustering.
 */
namespace kmeans {

/**
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 *
 * @param data A matrix containing data for each observation. 
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * On output, each column should contain the initial centroid location for its cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * On output, this will contain the 0-based cluster assignment for each observation, where each entry is less than `num_centers`.
 *
 * @return Details of the clustering, including the size of each cluster and the status of the algorithm.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
Details<Index_> compute(
    const Matrix_& data, 
    const Initialize<Index_, Data_, Cluster_, Float_, Matrix_>& initialize, 
    const Refine<Index_, Data_, Cluster_, Float_, Matrix_>& refine,
    Cluster_ num_centers,
    Float_* centers,
    Cluster_* clusters)
{
    auto actual_centers = initialize.run(data, num_centers, centers);
    auto output = refine.run(data, actual_centers, centers, clusters);
    sanisizer::resize(output.sizes, num_centers); // restoring the full size.
    return output;
}

/**
 * Overload of `compute()` to assist template deduction for the default `Matrix`.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * @tparam Data_ Numeric type of the input dataset.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 *
 * @param data A matrix containing data for each observation. 
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * On output, each column should contain the initial centroid location for its cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * On output, this will contain the 0-based cluster assignment for each observation, where each entry is less than `num_centers`.
 *
 * @return Details of the clustering, including the size of each cluster and the status of the algorithm.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_>
Details<Index_> compute(
    const Matrix<Index_, Data_>& data,
    const Initialize<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& initialize, 
    const Refine<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& refine,
    const Cluster_ num_centers,
    Float_* const centers,
    Cluster_* const clusters)
{
    return compute<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >(data, initialize, refine, num_centers, centers, clusters);
}

/**
 * @brief Results of the k-means clustering.
 */
template<typename Index_, typename Cluster_, typename Float_>
struct Results {
    /**
     * @cond
     */
    Results(const std::size_t num_dimensions, const Index_ num_observations, const Cluster_ num_centers) : 
        centers(sanisizer::product<decltype(I(centers.size()))>(num_dimensions, num_centers)),
        clusters(num_observations)
    {}

    Results() = default;
    /**
     * @endcond
     */

    /**
     * An array of length equal to the number of observations, containing the 0-indexed cluster assignment for each observation.
     * Each entry is less than the number of clusters.
     */
    std::vector<Cluster_> clusters;

    /**
     * An array of length equal to the product of the number of dimensions and clusters.
     * This contains a column-major matrix where each row corresponds to a dimension and each column corresponds to a cluster.
     * Each column contains the centroid coordinates for the associated cluster.
     */
    std::vector<Float_> centers;

    /**
     * Further details from running the chosen k-means algorithm.
     */
    Details<Index_> details;
};

/**
 * Overload of `compute()` that allocates and returns the vectors for the centroids and cluster assignments.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 *
 * @param data A matrix containing data for each observation. 
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 *
 * @return Results of the clustering, including the centroid locations and cluster assignments.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
Results<Index_, Cluster_, Float_> compute(
    const Matrix_& data, 
    const Initialize<Index_, Data_, Cluster_, Float_, Matrix_>& initialize, 
    const Refine<Index_, Data_, Cluster_, Float_, Matrix_>& refine,
    const Cluster_ num_centers)
{
    Results<Index_, Cluster_, Float_> output;
    sanisizer::resize(output.clusters, data.num_observations());
    output.centers.resize(sanisizer::product<decltype(I(output.centers.size()))>(num_centers, data.num_dimensions()));
    output.details = compute(data, initialize, refine, num_centers, output.centers.data(), output.clusters.data());
    return output;
}

/**
 * Overload of `compute()` to assist template deduction for the default `Matrix`.
 * This allocates and returns the vectors for the centroids and cluster assignments.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * @tparam Data_ Numeric type of the input dataset.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 *
 * @param data A matrix containing data for each observation. 
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 *
 * @return Results of the clustering, including the centroid locations and cluster assignments.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_>
Results<Index_, Cluster_, Float_> compute(
    const Matrix<Index_, Data_>& data,
    const Initialize<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& initialize, 
    const Refine<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& refine,
    const Cluster_ num_centers)
{
    return compute<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >(data, initialize, refine, num_centers);
}

}

#endif
