#ifndef KMEANS_REINITIALIZE_HPP
#define KMEANS_REINITIALIZE_HPP 

#include <vector>
#include <algorithm>
#include <cstdint>
#include "initialization.hpp"
#include "QuickSearch.hpp"

namespace kmeans {

class Reinitialize {
public:
    struct Defaults {
        static constexpr uint64_t seed = 1234u;

        static constexpr bool recompute_clusters = true;

        static constexpr int iterations = 10;
    };

private:
    uint64_t seed = Defaults::seed;
    bool recompute_clusters = Defaults::recompute_clusters;
    int iterations = Defaults::iterations;

public:
    Reinitialize& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }

    Reinitialize& set_recompute_clusters(bool r = Defaults::recompute_clusters) {
        recompute_clusters = r;
        return *this;
    }

    Reinitialize& set_iterations(int i = Defaults::iterations) {
        iterations = i;
        return *this;
    }

public:
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

