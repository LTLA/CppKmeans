#ifndef KMEANS_INITIALIZE_PCA_PARTITION_HPP
#define KMEANS_INITIALIZE_PCA_PARTITION_HPP

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "aarand/aarand.hpp"
#include "powerit/PowerIterations.hpp"
#include "Base.hpp"

/**
 * @file InitializePcaPartition.hpp
 *
 * @brief Class for k-means initialization with PCA partitioning.
 */
namespace kmeans {

/**
 * @brief Options for `InitializePcaPartition`.
 */
struct InitializePcaPartitionOptions {
    /**
     * Options to use for the power iterations.
     */
    powerit::Options power_iteration_options;

    /**
     * Size adjustment value, check out `InitializePcaPartition` for more details.
     * This value should lie in `[0, 1]`.
     */
    double size_adjustment = 1;

    /**
     * Random seed to use to construct the PRNG for the power iterations.
     */
    static constexpr uint64_t seed = 6523u;
};

/**
 * @cond
 */
namespace internal {

template<typename Center_>
struct PowerWorkspace {
    std::vector<Center_> pc;
    std::vector<Center_> delta;
    std::vector<Center_> cov;
};

template<class Matrix_, typename Center_, class Engine_>
std::vector<Data_> compute_pc1(
    const Matrix_& data, 
    const std::vector<typename Matrix_::index_type>& chosen, 
    const Center_* center, 
    Engine_& eng, 
    PowerWorkspace<Matrix_, Center_>& work,
    int nthreads,
    const powerit::Options& power_opts)
{
    auto ndim = data.num_observations();
    work.pc.resize(ndim);
    work.delta.resize(ndim);
    size_t long_ndim = ndim;
    work.cov.resize(long_ndim * long_ndim);

    work.matwork.clear();
    work.matwork.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        work.matwork.push_back(matrix.
    }

    // Computing the lower triangle of the covariance matrix. 
    auto work = data
    for (auto i : chosen) {
        auto dptr = data + i * ndim;
        for (int j = 0; j < ndim; ++j) {
            delta[j] = dptr[j] - center[j];
        }
        for (int j = 0; j < ndim; ++j) {
            for (int k = 0; k <= j; ++k) {
                cov[j * ndim + k] += delta[j] * delta[k];
            }
        }
    }

    // Filling in the other side of the matrix, to enable cache-efficient multiplication.
    for (int j = 0; j < ndim; ++j) {
        for (int k = j + 1; k < ndim; ++k) {
            cov[j * ndim + k] = cov[k * ndim + j];
        }
    }

    powerit::compute(ndim, cov.data(), work.pc.data(), eng, power_opts);
    return output;
} 

}
/**
 * @endcond
 */

/**
 * @brief Implements the PCA partitioning method of Su and Dy (2007).
 *
 * This approach involves the selection of starting points via iterative partitioning based on principal components analysis.
 * The aim is to obtain well-separated starting points for refinement with algorithms like Hartigan-Wong or Lloyd.
 * This is achieved by selecting the most dispersed cluster for further partitioning.
 *
 * We start from a single cluster containing all points.
 * At each iteration, we select the cluster with the largest within-cluster sum of squares (WCSS);
 * we identify the first principal component within that cluster;
 * and we split the cluster at its center along that axis to obtain two new clusters.
 * This is repeated until the desired number of clusters is obtained, and the centers and cluster identifiers are then reported.
 *
 * The original algorithm favors selection and partitioning of the largest cluster, which has the greatest chance of having the highest WCSS.
 * For more fine-grained control, we modify the procedure to adjust the effective number of observations contributing to the WCSS.
 * Specifically, we choose the cluster to partition based on the product of $N$ and the mean squared difference within each cluster,
 * where $N$ is the cluster size raised to some user-specified power (i.e., the "size adjustment") between 0 and 1.
 * An adjustment of 1 recapitulates the original algorithm, while smaller values of the size adjustment will reduce the preference towards larger clusters.
 * A value of zero means that the cluster size is completely ignored, though this seems unwise as it causes excessive splitting of small clusters with unstable WCSS.
 *
 * This method is not completely deterministic as a randomization step is used in the PCA.
 * Nonetheless, the stochasticity is likely to have a much smaller effect than in the other initialization methods.
 *
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster index.
 * @tparam Index_ Integer type for the observation index.
 *
 * @see
 * Su, T. and Dy, J. G. (2007).
 * In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
 * _Intelligent Data Analysis_ 11, 319-338.
 */
template<typename Data_ = double, typename Cluster_ = int, typename Index_ = int>
class InitializePcaPartition : public Initialize<Data_, Cluster_, Index_> {
public:
    /**
     * @param options Options for PCA partitioning.
     */
    InitializePcaPartition(InitializePcaPartition<Data_> options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializePcaPartition() = default;

private:
    InitializePcaPartitionOptions my_options;

public:

    static void compute_center(int ndim, Index_ nobs, const Data_* data, Data_* center) {
        std::fill(center, center + ndim, 0);
        for (Index_ i = 0; i < nobs; ++i) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                center[d] += dptr[d];
            }
        }
        for (int d = 0; d < ndim; ++d) {
            center[d] /= nobs;
        }
    }

    static void compute_center(int ndim, const std::vector<Index_>& chosen, const Data_* data, Data_* center) {
        std::fill(center, center + ndim, 0);
        for (auto i : chosen) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                center[d] += dptr[d];
            }
        }
        for (int d = 0; d < ndim; ++d) {
            center[d] /= chosen.size();
        }
    }

    static Data_ update_mrse(int ndim, const std::vector<Index_>& chosen, const Data_* data, Data_* center) {
        compute_center(ndim, chosen, data, center);

        Data_ curmrse = 0;
        for (auto i : chosen) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                Data_ delta = dptr[d] - center[d];
                curmrse += delta * delta;
            }
        }

        return curmrse / chosen.size();
    }
    /**
     * @endcond
     */
public:
    /*
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param data Pointer to an array where the dimensions are rows and the observations are columns.
     * Data should be stored in column-major format.
     * @param ncenters Number of centers to pick.
     * @param[out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On output, this will contain the final centroid locations for each cluster.
     * Data should be stored in column-major order.
     * @param clusters Pointer to an array of length `nobs`.
     * This is used as a buffer and the contents on output should not be used.
     *
     * @return `centers` is filled with the new cluster centers.
     * The number of filled centers is returned, see `Initializer::run()`.
     */
    Cluster_ run(int ndim, Index_ nobs, const Data_* data, Cluster_ ncenters, Data_* centers, Cluster_* clusters) {
        if (nobs == 0) {
            return 0;
        }

        std::mt19937_64 rng(seed);
        std::vector<Data_> mrse(ncenters);
        std::vector<std::vector<Index_> > assignments(ncenters);

        // Setting up the zero'th cluster. (No need to actually compute the
        // MRSE at this point, as there's nothing to compare it to.)
        compute_center(ndim, nobs, data, centers);
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);
        std::fill(clusters, clusters + nobs, 0);

        for (Cluster_ cluster = 1; cluster < ncenters; ++cluster) {
            Data_ worst_ss = 0;
            Index_ worst_cluster = 0;
            for (Cluster_ i = 0; i < cluster; ++i) {
                Data_ multiplier = assignments[i].size();
                if (adjust != 1) {
                    multiplier = std::pow(static_cast<Data_>(multiplier), adjust);
                }

                Data_ pseudo_ss = mrse[i] * multiplier;
                if (pseudo_ss > worst_ss) {
                    worst_ss = pseudo_ss;
                    worst_cluster = i;
                }
            }

            // Extracting the principal component for this bad boy.
            auto worst_center = centers + worst_cluster * ndim;
            auto& worst_assignments = assignments[worst_cluster];
            auto pc1 = compute_pc1(ndim, worst_assignments, data, worst_center, rng);

            // Projecting all points in this cluster along PC1. The center lies
            // at zero, so everything positive (on one side of the hyperplane
            // orthogonal to PC1 and passing through the center) gets bumped to
            // the next cluster.
            std::vector<Index_>& new_assignments = assignments[cluster];
            std::vector<Index_> worst_assignments2;
            for (auto i : worst_assignments) {
                auto dptr = data + i * ndim;
                Data_ proj = 0;
                for (int d = 0; d < ndim; ++d) {
                    proj += (dptr[d] - worst_center[d]) * pc1[d];
                }

                if (proj > 0) {
                    new_assignments.push_back(i);
                } else {
                    worst_assignments2.push_back(i);
                }
            }

            // If one or the other is empty, then this entire procedure short
            // circuits as all future iterations will just re-select this
            // cluster (which won't get partitioned properly anyway). In the
            // bigger picture, the quick exit out of the iterations is correct
            // as we should only fail to partition in this manner if all points
            // within each remaining cluster are identical.
            if (new_assignments.empty() || worst_assignments2.empty()) {
                return cluster;
            }

            for (auto i : new_assignments) {
                clusters[i] = cluster;
            }
            worst_assignments.swap(worst_assignments2);

            // Computing centers and MRSE.
            auto new_center = centers + cluster * ndim;
            mrse[cluster] = update_mrse(ndim, new_assignments, data, new_center);
            mrse[worst_cluster] = update_mrse(ndim, worst_assignments, data, worst_center);
        }

        return ncenters;
    }
};

}

#endif
