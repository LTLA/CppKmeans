#ifndef KMEANS_INITIALIZE_PCA_PARTITION_HPP
#define KMEANS_INITIALIZE_PCA_PARTITION_HPP

#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "aarand/aarand.hpp"
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
        iters = i;
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
        observation_cap = c;
        return *this;
    }

    /**
     * @param Random seed to use to construct the PRNG for the power method.
     * @return A reference to this `InitializePCAPartition` object.
     */
    InitializePCAPartition& set_seed(uint64_t s = Defaults::seed) {
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
        for (size_t j = 0; j < ndim; ++j) {
            for (size_t k = j + 1; k < ndim; ++k) {
                cov[j * ndim + k] = cov[k * ndim + j];
            }
        }

        // Defining a random starting vector.
        std::vector<DATA_t> output(ndim); 
        while (1) {
            for (int d = 0; d < ndim - 1; d += 2) {
                auto sampled = aarand::standard_normal<DATA_t>(eng);
                output[d] = sampled.first;
                output[d + 1] = sampled.second;
            }
            if (ndim % 2) {
                output.back() = aarand::standard_normal<DATA_t>(eng).first;
            }
            if (normalize(ndim, output.data())) {
                break;
            }
        }

        // Applying power iterations.
        for (int i = 0; i < iters; ++i) {
            for (size_t j = 0; j < ndim; ++j) {
                // As the matrix is symmetric, we can use inner_product.
                // This technically computes the transpose of the matrix
                // with the vector, but it's all the same, so whatever.
                delta[j] = std::inner_product(output.begin(), output.end(), cov.data() + j * ndim, static_cast<DATA_t>(0.0));
            }

            // Normalizing the matrix.
            auto l2 = normalize(ndim, delta.data());

            // We want to know if SIGMA * output = lambda * output, i.e., l2 * delta = lambda * output.
            // If we use l2 as a working estimate for lambda, we're basically just testing the difference
            // between delta and output. We compute the error and compare this to the tolerance.
            DATA_t err = 0;
            for (int d = 0; d < ndim; ++d) {
                DATA_t diff = delta[d] - output[i];
                err += diff * diff;
            }
            if (std::sqrt(err) < tol) {
                break;
            }

            std::copy(delta.begin(), delta.end(), output.begin());
        }

        return output;
    } 

    static DATA_t update_mrse(int ndim, const std::vector<INDEX_t>& chosen, const DATA_t* data, DATA_t* center) {
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
        std::fill(clusters, clusters + nobs, 0);
        std::fill(centers, centers + ndim, 0);
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
            for (CLUSTER_t i = 0; i < cluster; ++i) {
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

            // Computing centers and MRSE.
            if (cluster + 1 < ncenters) {
                if (new_assignments.size()) {
                    auto new_center = centers + cluster * ndim;
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
