#ifndef KMEANS_REINITIALIZE_HPP
#define KMEANS_REINITIALIZE_HPP 

#include <vector>
#include <algorithm>
#include "initialization.hpp"
#include "QuickSearch.hpp"

namespace kmeans {

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
void reinitialize_centers(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters, ENGINE& eng, bool empty_clusters = true) {
    std::vector<DATA_t> mindist(nobs);
    std::vector<DATA_t> cumulative(nobs);

    if (empty_clusters) {
        QuickSearch x(ndim, ncenters, centers);
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            auto closest = x.find_with_distance(data + obs * ndim);
            clusters[obs] = closest.first;
            mindist[obs] = closest.second * closest.second;
        }
    } else {
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            const DATA_t* o = data + ndim * obs;
            const DATA_t* c = centers + ndim * clusters[obs];
            auto& current = mindist[obs];
            for (int d = 0; d < ndim; ++d) {
                current += (o[d] - c[d]) * (o[d] - c[d]);
            }
        }
    }

    for (CLUSTER_t c = 0; c < ncenters; ++c) {
        // Erasing cluster 'c' and search on the rest. For c > 0, this erasure
        // is implicitly handled by the replacement with the sampled point.
        QuickSearch x(ndim, ncenters - 1, centers + ndim);
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            if (clusters[obs] == c) {
                auto closest = x.find_with_distance(data + obs * ndim);
                mindist[obs] = closest.second * closest.second;

                // There is a frameshift on the cluster identities due to our
                // removal and iterative replacement of cluster 'c'.  Thus, we
                // need to adjust the cluster IDs to ensure that they do not 
                // conflict with the original cluster identities, given that
                // we need the originals for the later updates.
                if (closest.first < c) {
                    clusters[obs] = closest.first;
                } else {
                    clusters[obs] = closest.first + 1;
                }
            } 
        }

        cumulative[0] = mindist[0];
        for (INDEX_t i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }
        auto chosen_id = weighted_sample(cumulative, mindist, nobs, eng);
        auto chosen_ptr = data + chosen_id * ndim;

        // Updating the minimum distances and the closest clusters.
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            const DATA_t* o = data + ndim * obs;
            DATA_t candidate = 0;
            for (int d = 0; d < ndim; ++d) {
                candidate += (chosen_ptr[d] - o[d]) * (chosen_ptr[d] - o[d]);
            }
            if (candidate < mindist[obs]) {
                mindist[obs] = candidate;
                clusters[obs] = c;
            }
        }

        if (c + 1 < ncenters) {
            // Implicit erasure happens here when we replace the center for the
            // next 'c' with the current sampled point. Note that we also update
            // the cluster assignments for good measure.
            auto target_ptr = centers + (c + 1) * ndim;
            std::copy(chosen_ptr, chosen_ptr + ndim, target_ptr);
        } else {
            // Undoing the frameshift on the cluster identities.
            for (CLUSTER_t c = 0; c < ncenters - 1; ++c) {
                auto dest = centers + c * ndim;
                auto src = dest + ndim;
                std::copy(src, src + ndim, dest);
            }
            std::copy(chosen_ptr, chosen_ptr + ndim, centers + c * ndim);
        }
    }

    return;
}

}

#endif

