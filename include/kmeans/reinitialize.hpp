#ifndef KMEANS_REINITIALIZE_HPP
#define KMEANS_REINITIALIZE_HPP 

#include <vector>
#include <algorithm>

namespace kmeans {

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
void reinitialize (int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters, bool empty_clusters = true) {
    std::vector<DATA_t> mindist(ndim);
    std::vector<DATA_t> cumulative(nobs);

    if (empty_clusters) {
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            auto closest = x.find_with_distance(data + obs * ndim);
            clusters[obs] = closest.first;
            mindist[obs] = closest.second * closest.second;
        }
    } else {
        #pragma omp parallel for
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            DATA_t* o = data + ndim * obs;
            DATA_t* c = centers + ndim * clusters[obs];
            auto& current = mindist[obs];
            for (int d = 0; d < ndim; ++d) {
                current += d * d;
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
        const auto total = cumulative.back();

        // See initialize.hpp for the logic around using a loop here.
        INDEX_t chosen_id = 0;
        do {
            const DATA_t sampled_weight = total * aarand::standard_uniform(eng);
            chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();
        } while (chosen_id == nobs || mindist[chosen_id] == 0);

        if (c + 1 < ncenters) {
            // Implicit erasure happens here when we replace the center for the
            // next 'c' with the current sampled point. Note that we also update
            // the cluster assignments for good measure.
            auto chosen_ptr = data + chosen_id * ndim;
            auto target_ptr = centers + (c + 1) * ndim;
            std::copy(chosen_ptr, chosen_ptr + ndim, target_ptr);

            #pragma omp parallel for
            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                DATA_t* o = data + ndim * obs;
                DATA_t candidate = 0;
                for (int d = 0; d < ndim; ++d) {
                    candidate += (chosen_ptr[d] - o[d]) * (chosen_ptr[d] - o[d]);
                }
                if (candidate < mindist[obs]) {
                    mindist[obs] = candidate;
                    clusters[obs] = c;
                }
            }
        } else {
            // Undoing the frameshift on the cluster identities.
            for (CLUSTER_t c = 0; c < ncenters - 1; ++c) {
                auto dest = centers + c * ndim;
                auto src = dest + ndim;
                std::copy(src, src + ndim, dest);
            }

            auto chosen_ptr = data + chosen_id * ndim;
            std::copy(chosen_ptr, chosen_ptr + ndim, centers + c * ndim);
        }
    }

    return;
}

}

#endif

