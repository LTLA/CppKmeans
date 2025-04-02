#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

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
 *
 * @brief Implements the full k-means clustering procedure.
 */

/**
 * @namespace kmeans
 * @brief Namespace for k-means clustering.
 */
namespace kmeans {

/**
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
 *
 * @param data A matrix-like object containing per-observation data.
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * On output, each column should contain the initial centroid location for its cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * On output, this will contain the 0-based cluster assignment for each observation.
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
    output.sizes.resize(num_centers); // restoring the full size.
    return output;
}

/**
 * Overload of `compute()` to assist template deduction for the default `Matrix`.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 *
 * @param data A matrix-like object containing per-observation data.
 * @param initialize Initialization method to use.
 * @param refine Refinement method to use.
 * @param num_centers Number of cluster centers.
 * @param[out] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * On output, each column should contain the initial centroid location for its cluster.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * On output, this will contain the 0-based cluster assignment for each observation.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_>
Details<Index_> compute(
    const Matrix<Index_, Data_>& data,
    const Initialize<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& initialize, 
    const Refine<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >& refine,
    Cluster_ num_centers,
    Float_* centers,
    Cluster_* clusters)
{
    return compute<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >(data, initialize, refine, num_centers, centers, clusters);
}

/**
 * @brief Full statistics from k-means clustering.
 */
template<typename Index_, typename Cluster_, typename Float_>
struct Results {
    /**
     * @cond
     */
    template<typename Dim_>
    Results(Dim_ num_dimensions, Index_ num_observations, Cluster_ num_centers) : 
        centers(num_dimensions * num_centers), clusters(num_observations) {}

    Results() = default;
    /**
     * @endcond
     */

    /**
     * An array of length equal to the number of observations, containing 0-indexed cluster assignments for each observation.
     */
    std::vector<Cluster_> clusters;

    /**
     * An array containing a column-major matrix where each row corresponds to a dimension and each column corresponds to a cluster.
     * Each column contains the centroid coordinates for its cluster.
     */
    std::vector<Float_> centers;

    /**
     * Further details from the chosen k-means algorithm.
     */
    Details<Index_> details;
};

/**
 * Overload of `compute()` that allocates and returns the vectors for the centroids and cluster assignments.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
 *
 * @param data A matrix-like object containing per-observation data.
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
    Cluster_ num_centers)
{
    Results<Index_, Cluster_, Float_> output;
    output.clusters.resize(data.num_observations());
    output.centers.resize(static_cast<size_t>(num_centers) * static_cast<size_t>(data.num_dimensions()));
    output.details = compute(data, initialize, refine, num_centers, output.centers.data(), output.clusters.data());
    return output;
}

/**
 * Overload of `compute()` to assist template deduction for the default `Matrix`.
 * This allocates and returns the vectors for the centroids and cluster assignments.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 *
 * @param data A matrix-like object containing per-observation data.
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
    Cluster_ num_centers)
{
    return compute<Index_, Data_, Cluster_, Float_, Matrix<Index_, Data_> >(data, initialize, refine, num_centers);
}

}

#endif
