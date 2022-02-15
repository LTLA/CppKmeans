#ifndef KMEANS_INITIALIZE_PCA_PARTITION_HPP
#define KMEANS_INITIALIZE_PCA_PARTITION_HPP

#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "aarand/aarand.hpp"

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
 * To ensure that we do not just select the largest cluster (which natually would have a higher WCSS),
 * we modify the procedure slightly to cap the effective number of observations contributing to the WCSS.
 * Specifically, we choose the cluster to partition based on the product of $N$ and the mean squared difference within each cluster,
 * where $N$ is the smaller of the cluster size and an observation cap.
 * This means that, past the cap, the dispersion of the clusters is the only metric that matters for partitioning.
 *
 * The observation cap represents the point at which a cluster has enough observations such that its existence is not in doubt.
 * If the cap is set to 1, the partitioning will be done on the mean squared difference;
 * this risks excessive splitting of small clusters with highly variable WCSS.
 * An appropriate value for this cap depends on the context - for example, scRNA-seq applications might set the cap to 100 cells.
 * Setting the cap to zero will cause it to be ignored such that the original PCA partitioning method is performed.
 *
 * This method is not completely deterministic as a randomization step is used in the PCA.
 * Nonetheless, the stochasticity is likely to have a much smaller effect than in the other initialization methods.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam INDEX_t Integer type for the observation index.
 *
 * @seealso
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
         * See `set_iterations()` for more details.
         */
        static constexpr int iterations = 500;

        /**
         * See `set_tolerance()` for more details.
         */
        static constexpr DATA_t tolerance = 0.000001;

        /**
         * See `set_cap()` for more details.
         */
        static constexpr INDEX_t cap = 0;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 6523u;
    };

public:
    /**
     * @param i Maximum number of iterations of the power method to perform when finding the first PC.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_iterations(int i = Defaults::iterations) {
        iterations = i;
        return *this;
    }

    /**
     * @param t Convergence threshold to terminate the power method.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_tolerance(DATA_t t = Defaults::tolerance) {
        tol = t;
        return *this;
    }

    /**
     * @param c Cap on the number of observations.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_cap(INDEX_t c = Defaults::cap) {
        cap = c;
        return *this;
    }

    /**
     * @param Random seed to use to construct the PRNG for the power method.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializeKmeansPP& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }
private:
    int iters = Defaults::iterations;
    DATA_t tol = Defaults::tolerance;
    INDEX_t observation_cap = Defaults::cap;
    uint64_t seed = Defaults::seed;

public:
    /**
     * @cond
     */
    template<class Rng>
    std::vector<DATA_t> compute_pc1(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, const DATA_t* center, Rng& eng) {
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
                    buffer[j * ndim + k] += delta[j] * delta[k];
                }
            }
        }

        // Filling in the other side of the matrix, to enable cache-efficient multiplication.
        for (size_t j = 0; j < ndim; ++j) {
            for (size_t k = j + 1; k < ndim; ++k) {
                buffer[j * ndim + k] = buffer[k * ndim + j];
            }
        }

        // Defining a random starting vector.
        std::vector<DATA_t> output(ndim); 
        for (int d = 0; d < ndim - 1; d += 2) {
            auto sampled = aarand::standard_normal<DATA_t>(rng);
            output[d] = sampled.first;
            output[d + 1] = sampled.second;
        }
        if (ndim % 2) {
            output.back() = aarand::standard_normal<DATA_t>(rng).first;
        }

        // Applying power iterations.
        for (int i = 0; i < iters; ++i) {
            for (size_t j = 0; j < ndim; ++j) {
                // As the matrix is symmetric, we can use inner_product.
                // This technically computes the transpose of the matrix
                // with the vector, but it's all the same, so whatever.
                delta[j] = std::inner_product(output.begin(), output.end(), buffer.data() + j * ndim, static_cast<DATA_t>(0.0));
            }

            // Normalizing the matrix.
            DATA_t ss = 0;
            for (auto d : delta) { 
                ss += d * d;
            }
            ss = std::sqrt(ss);
            if (!ss) {
                break;
            }
            for (auto& d : delta) {
                d /= ss;
            }

            // Comparing to the previous value.
            auto diff = std::inner_product(delta.begin(), delta.end(), output.begin(), static_cast<DATA_t>(0.0));
            if (std::abs(diff - 1) < tol) {
                break;
            }

            std::copy(delta.begin(), delta.end(), output.begin());
        }

        return output;
    } 

    DATA_t update_mrse(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, DATA_t* center) {
        for (auto i : chosen) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                center[d] += dptr[d];
            }
        }

        for (int d = 0; d < ndim; ++d) {
            center[d] /= chosen.size();
        }

        DATA_t curmrse = 0;
        for (auto i : chosen) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                DATA_t delta = dptr[d] - center[d];
                curmrse += delta * delta;
            }
        }

        return curmrse / chosens.size();
    }
    /**
     * @endcond
     */
public:
    CLUSTER_t run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        if (nobs == 0) {
            return 0;
        }

        std::mt19937_64 rng(seed);
        std::vector<DATA_t> mrse(ncenters);
        std::vector<std::vector<INDEX_t> > assignments(ncenters);

        // Setting up the zero'th cluster. (No need to actually compute the
        // MRSE at this point, as there's nothing to compare it to.)
        std::fill(clusters, clusters + nobs, 0);
        std::fill(centers, ncenters + ndim, 0);
        for (size_t i = 0; i < nobs; ++i) {
            auto dptr = data + i * ndim;
            for (int d = 0; d < ndim; ++d) {
                centers[d] += dptr[d];
            }
        }
        for (int d = 0; d < ndim; ++d) {
            centers[d] /= nobs;
        }
        assignments[0].resize(nobs);
        std::iota(assignments.front().begin(), assignments.front().end(), 0);

        for (CLUSTER_t cluster = 1; cluster < ncenters; ++cluster) {
            // Choosing the cluster with the largest within-cluster SS. Here we
            // apply an observation cap to avoid favoring large clusters. We
            // don't use the mean SS as this may encourage excessive
            // partitioning of small clusters with unstable mean SS values.
            DATA_t worst_ss = 0;
            INDEX_t worst_cluster = 0;
            for (size_t i = 0; i < counts.size(); ++i) {
                DATA_t multiplier = assignments[i].size();
                if (observation_cap && static_cast<DATA_t>(observation_cap) < multiplier) {
                    multiplier = observation_cap;
                }

                DATA_t pseudo_ss = mrse[i] * multiplier;
                if (pseudo_ss > worst_ss) {
                    worst_ss = pseudo_ss;
                    worst_cluster = i;
                }
            }

            // Extracting the principal component for this bad boy.
            auto worst_center = center + worst_cluster * ndim;
            auto pc1 = compute_pc1(ndim, assignments[worst_cluster], data, clusters, worst_cluster, worst_center, rng);

            // Projecting all points in this cluster along PC1. The center lies
            // at zero, so everything positive (on one side of the hyperplane
            // orthogonal to PC1 and passing through the center) gets bumped to
            // the next cluster.
            auto& worst_assignments = assignments[worst_cluster];
            std::vector<INDEX_t>& new_assignments = assignments[fulfilled];
            std::vector<INDEX_t> worst_assignments2;
            for (auto i : worst_assignments) {
                auto dptr = data + i * ndim;
                DATA_t proj = 0;
                for (int j = 0; j < ndim; ++j) {
                    proj += (dptr[j] - worst_center[j]) * pc1[j];
                }

                if (proj > 0) {
                    new_assignments.push_back(i);
                } else {
                    worst_assignments2.push_back(i);
                }
            }

            // If one or the other is empty, then this entire procedure short
            // circuits as all future iterations will just re-select this
            // cluster (which won't get partitioned properly anyway). The
            // quick return is correct as we would only fail to partition
            // if all points are identical.
            if (new_assignments.empty() || worst_assignments2.empty()) {
                return cluster;
            }

            for (auto i : new_assignments) {
                clusters[i] = cluster;
            }
            worst_assignments.swap(worst_assignments2);
            ++fulfilled;

            // Computing centers and MRSE.
            if (cluster + 1 < nclusters) {
                if (new_assignments.size()) {
                    auto new_center = center + cluster * ndim;
                    update_mrse(ndim, new_assignments, data, new_center);
                }
                if (worst_assignments.size()) {
                    std::fill(worst_center, worst_center + ndim, 0);
                    update_mrse(ndim, worst_assignments, data, worst_center);
                }
            }
        }

        return ncenters;
    }
};

}

#endif
