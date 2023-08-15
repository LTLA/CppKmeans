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
 * @file InitializePCAPartition.hpp
 *
 * @brief Class for k-means initialization with PCA partitioning.
 */
namespace kmeans {

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
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam INDEX_t Integer type for the observation index.
 *
 * @see
 * Su, T. and Dy, J. G. (2007).
 * In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
 * _Intelligent Data Analysis_ 11, 319-338.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class InitializePCAPartition : public Initialize<DATA_t, CLUSTER_t, INDEX_t> {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_size_adjustment()` for more details.
         */
        static constexpr DATA_t size_adjustment = 1;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 6523u;
    };

public:
    /**
     * @param i Maximum number of power iterations to use in `powerit::PowerIterations::set_iterations()`.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_iterations(int i = powerit::PowerIterations::Defaults::iterations) {
        iters = i;
        return *this;
    }

    /**
     * @param t Convergence threshold to use in `powerit::PowerIterations::set_tolerance()`.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_tolerance(DATA_t t = powerit::PowerIterations::Defaults::tolerance) {
        tol = t;
        return *this;
    }

    /**
     * @param s Size adjustment value, should lie in [0, 1].
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_size_adjustment(DATA_t s = Defaults::size_adjustment) {
        adjust = s;
        return *this;
    }

    /**
     * @param s Random seed to use to construct the PRNG for the power method.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }
private:
    int iters = powerit::PowerIterations::Defaults::iterations;
    DATA_t tol = powerit::PowerIterations::Defaults::tolerance;
    DATA_t adjust = Defaults::size_adjustment;
    uint64_t seed = Defaults::seed;

public:
    /**
     * @cond
     */
    static DATA_t normalize(int ndim, DATA_t* x) {
        DATA_t ss = 0;
        for (int d = 0; d < ndim; ++d) {
            ss += x[d] * x[d];
        }

        if (ss) {
            ss = std::sqrt(ss);
            for (int d = 0; d < ndim; ++d) {
                x[d] /= ss;
            }
        }
        return ss;
    }

    template<class Rng>
    std::vector<DATA_t> compute_pc1(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, const DATA_t* center, Rng& eng) const {
        std::vector<DATA_t> delta(ndim);
        std::vector<DATA_t> cov(ndim * ndim);

        // Computing the lower triangle of the covariance matrix. 
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

        powerit::PowerIterations power;
        power.set_iterations(iters).set_tolerance(tol);

        std::vector<DATA_t> output(ndim);
        power.run(ndim, cov.data(), output.data(), eng);
        return output;
    } 

    static void compute_center(int ndim, INDEX_t nobs, const DATA_t* data, DATA_t* center) {
        std::fill(center, center + ndim, 0);
        for (INDEX_t i = 0; i < nobs; ++i) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                center[d] += dptr[d];
            }
        }
        for (int d = 0; d < ndim; ++d) {
            center[d] /= nobs;
        }
    }

    static void compute_center(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, DATA_t* center) {
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

    static DATA_t update_mrse(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, DATA_t* center) {
        compute_center(ndim, chosen, data, center);

        DATA_t curmrse = 0;
        for (auto i : chosen) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                DATA_t delta = dptr[d] - center[d];
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
    CLUSTER_t run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        if (nobs == 0) {
            return 0;
        }

        std::mt19937_64 rng(seed);
        std::vector<DATA_t> mrse(ncenters);
        std::vector<std::vector<INDEX_t> > assignments(ncenters);

        // Setting up the zero'th cluster. (No need to actually compute the
        // MRSE at this point, as there's nothing to compare it to.)
        compute_center(ndim, nobs, data, centers);
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);
        std::fill(clusters, clusters + nobs, 0);

        for (CLUSTER_t cluster = 1; cluster < ncenters; ++cluster) {
            DATA_t worst_ss = 0;
            INDEX_t worst_cluster = 0;
            for (CLUSTER_t i = 0; i < cluster; ++i) {
                DATA_t multiplier = assignments[i].size();
                if (adjust != 1) {
                    multiplier = std::pow(static_cast<DATA_t>(multiplier), adjust);
                }

                DATA_t pseudo_ss = mrse[i] * multiplier;
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
            std::vector<INDEX_t>& new_assignments = assignments[cluster];
            std::vector<INDEX_t> worst_assignments2;
            for (auto i : worst_assignments) {
                auto dptr = data + i * ndim;
                DATA_t proj = 0;
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
