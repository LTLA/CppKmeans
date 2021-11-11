#ifndef KMEANS_REINITIALIZE_HPP
#define KMEANS_REINITIALIZE_HPP 

#include <vector>
#include <algorithm>
#include <cstdint>
#include "initialization.hpp"
#include "QuickSearch.hpp"

/**
 * @file reinitialize.hpp
 * 
 * @brief Reinitialize from an existing set of clusters.
 */

namespace kmeans {

/**
 * @brief Reinitialize from an existing set of clusters.
 *
 * Imagine that we have a high-quality clustering of a dataset that we wish to update with more observations.
 * The goal is to create a new set of clusters that (i) re-uses information from the existing "good" clustering for the previous observations
 * while (ii) accommodating distinct clusters that are unique to the set of new observations.
 * We cannot simply rely on the usual k-means iterations to create a new cluster as the bulk of previous observations discourage large movements of the centers.
 * On the other hand, rerunning the k-means from scratch will discard all information from the existing clustering,
 * possibly reducing the solution quality - especially if the previous clustering was already finely optimized, e.g., via restarts or high iteration counts.
 *
 * This class implements a reinitialization strategy for an existing k-means clustering after adding new observations.
 * We remove each cluster center in turn and use a **kmeans++** approach to randomly propose a new cluster center.
 * Sampling probabilities for all observations are weighted by the distances from the other (non-removed) centers. 
 * If the proposed cluster center achieves a lower within-cluster-sum-squares (WCSS) than the clustering before removal of the previous center, 
 * then the update is accepted and the proposed center becomes the new cluster center.
 * This is repeated for each center and the updated set of cluster centers is reported for further refinement with, e.g., `HartiganWong` or `Lloyd`.
 *
 * Our strategy favors the preservation of existing clusters if there are no better arrangements. 
 * However, it can still respond to the presence of new clusters by discarding an existing center if the distance (and thus sampling probability) is large enough. 
 * We can reduce the strength of this preference for preservation by repeating the sampling across several iterations, 
 * increasing the chance of finding an alternative center with a lower WCSS.
 */
class Reinitialize {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 1234u;

        /**
         * See `set_recompute_clusters()` for more details.
         */
        static constexpr bool recompute_clusters = true;

        /**
         * See `set_iterations()` for more details.
         */
        static constexpr int iterations = 10;
    };

private:
    uint64_t seed = Defaults::seed;
    bool recompute_clusters = Defaults::recompute_clusters;
    int iterations = Defaults::iterations;

public:
    /**
     * @param s Random seed for the PRNG.
     *
     * @return A reference to this `Reinitialize` object.
     */ 
    Reinitialize& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param r Whether to compute the closest cluster for each observation on input.
     * If `false`, we assume that the identity of the closest cluster is already provided in the `clusters` array in `run()`.
     * This can avoid a redundant search if the cluster is already known, e.g., from previous k-means iterations.
     * 
     * @return A reference to this `Reinitialize` object.
     */
    Reinitialize& set_recompute_clusters(bool r = Defaults::recompute_clusters) {
        recompute_clusters = r;
        return *this;
    }

    /**
     * @param i Number of iterations to attempt to find a new center with a lower WCSS than the previous clustering. 
     * Larger values increase the likelihood that the cluster centers will be changed during reinitialization.
     *
     * @return A reference to this `Reinitialize` object.
     */
    Reinitialize& set_iterations(int i = Defaults::iterations) {
        iterations = i;
        return *this;
    }

public:
    /**
     * @tparam DATA_t Floating-point type for the data and centroids.
     * @tparam CLUSTER_t Integer type for the cluster assignments.
     * @tparam INDEX_t Integer type for the observation index.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On input, this should contain the previous centroid locations for each cluster.
     * Data should be stored in column-major order.
     * On output, this will contain the reinitialized centroid locations for each cluster.
     * @param[in, out] clusters Pointer to an array of length `nobs`.
     * On input, this should contain the identity of the closest cluster if `set_recompute_clusters()` is set to `false`; otherwise the input values are ignored.
     * On output, this will contain the (0-indexed) cluster assignment for each observation.
     *
     * @return `centers` and `clusters` are filled with the new centers and cluster assignments.
     */
    template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int>
    void run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) const {
        std::vector<DATA_t> mindist(nobs);
        std::vector<DATA_t> cumulative(nobs);

        if (recompute_clusters) {
            QuickSearch x(ndim, ncenters, centers);
            #pragma omp parallel for
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                clusters[obs] = x.find(data + obs * ndim);
            }
        }

        // Technically, we could get the distances from QuickSearch,
        // but this gets passaged through a sqrt(), so we just recompute
        // it to avoid problems with loss of numerical precision when
        // computing WCSS (which may cause headaches with equality
        // comparisons during the random sampling).
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            const DATA_t* o = data + ndim * obs;
            const DATA_t* c = centers + ndim * clusters[obs];
            auto& current = mindist[obs];
            for (int d = 0; d < ndim; ++d) {
                double delta = o[d] - c[d];
                current += delta * delta;
            }
        }

        std::mt19937_64 eng(seed);
        DATA_t wcss = std::accumulate(mindist.begin(), mindist.end(), 0.0);
        std::vector<CLUSTER_t> clusters_erased(nobs);
        std::vector<DATA_t> mindist_erased(nobs);

        for (CLUSTER_t c = 0; c < ncenters; ++c) {
            std::copy(mindist.begin(), mindist.end(), mindist_erased.begin());
            std::copy(clusters, clusters + nobs, clusters_erased.begin());

            // Erasing cluster 'c' and search on the rest. For c > 0, this erasure
            // is implicitly handled by the replacement with the sampled point.
            QuickSearch x(ndim, ncenters - 1, centers + ndim);
            #pragma omp parallel for
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                if (clusters_erased[obs] == c) {
                    auto closest = x.find_with_distance(data + obs * ndim);
                    mindist_erased[obs] = closest.second * closest.second;

                    // There is a frameshift on the cluster identities due to our
                    // removal and iterative replacement of cluster 'c'.  Thus, we
                    // need to adjust the cluster IDs to ensure that they do not 
                    // conflict with the original cluster identities, given that
                    // we need the originals for the later updates.
                    if (closest.first < c) {
                        clusters_erased[obs] = closest.first;
                    } else {
                        clusters_erased[obs] = closest.first + 1;
                    }
                } 
            }

            cumulative[0] = mindist_erased[0];
            for (INDEX_t i = 1; i < nobs; ++i) {
                cumulative[i] = cumulative[i-1] + mindist_erased[i];
            }

            // Performing multiple iterations to choose a new cluster center,
            // making sure that the WCSS is lower than what it was before we
            // revoked the existing cluster center.
            std::vector<DATA_t> mindist_copy(nobs);
            std::vector<CLUSTER_t> clusters_copy(nobs);

            const DATA_t* chosen_ptr = NULL;
            for (int it = 0; it < iterations; ++it) {
                auto candidate = weighted_sample(cumulative, mindist_erased, nobs, eng);
                auto candidate_ptr = data + candidate * ndim;
                std::copy(mindist_erased.begin(), mindist_erased.end(), mindist_copy.begin());
                std::copy(clusters_erased.begin(), clusters_erased.end(), clusters_copy.begin());

                // Updating the minimum distances and the closest clusters.
                #pragma omp parallel for
                for (INDEX_t obs = 0; obs < nobs; ++obs) {
                    const DATA_t* o = data + ndim * obs;
                    DATA_t candidate_dist = 0;
                    for (int d = 0; d < ndim; ++d) {
                        double delta = candidate_ptr[d] - o[d];
                        candidate_dist += delta * delta;
                    }
                    if (candidate_dist < mindist_copy[obs]) {
                        mindist_copy[obs] = candidate_dist;
                        clusters_copy[obs] = c;
                    }
                }

                double wcss_copy = std::accumulate(mindist_copy.begin(), mindist_copy.end(), 0.0);
                if (wcss > wcss_copy) {
                    mindist.swap(mindist_copy);
                    std::copy(clusters_copy.begin(), clusters_copy.end(), clusters);
                    wcss = wcss_copy;
                    chosen_ptr = candidate_ptr;
                    break;
                }
            }

            if (c + 1 < ncenters) {
                // Implicit erasure happens here:
                // - The start of 'centers' contains the center of 'c' (possibly
                //   swapped in from a previous iteration).
                // - We swap it with the center of 'c + 1', so the buffer for
                //   'c + 1' now contains the center of 'c' and the start of 
                //   'centers' now contains the center for 'c + 1'.
                // - We optionally replace the buffer for 'c + 1' with the
                //   chosen point. Or not, if it wasn't good enough.
                auto target_ptr = centers + (c + 1) * ndim;
                for (int d = 0; d < ndim; ++d) {
                    std::swap(target_ptr[d], centers[d]);
                }
                if (chosen_ptr) {
                    std::copy(chosen_ptr, chosen_ptr + ndim, target_ptr);
                } 
            } else {
                // Undoing the frameshift on the cluster centers, given
                // that we effectively shifted everything to the right 
                // to allow a copy-free QuickSearch.
                std::vector<DATA_t> holding(centers, centers + ndim);
                for (CLUSTER_t c2 = 0; c2 < ncenters - 1; ++c2) {
                    auto dest = centers + c2 * ndim;
                    auto src = dest + ndim;
                    std::copy(src, src + ndim, dest);
                }

                auto final_ptr = centers + c * ndim;
                if (chosen_ptr) {
                    std::copy(chosen_ptr, chosen_ptr + ndim, final_ptr);
                } else {
                    std::copy(holding.begin(), holding.end(), final_ptr);
                }
            }
        }

        return;
    }
};

}

#endif

