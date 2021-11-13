#include <gtest/gtest.h>
#include "kmeans/Reinitialize.hpp"
#include "TestCore.h"

#include <vector>
#include <limits>

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
void ref_reinitialize_centers(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, ENGINE& eng) {
    std::vector<DATA_t> mindist(nobs);
    std::vector<DATA_t> cumulative(nobs);

    for (CLUSTER_t c = 0; c < ncenters; ++c) {
        std::vector<INDEX_t> clusters(nobs);
        
        // Finding the new closest cluster for each point.
        double wcss = 0;
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            const DATA_t* pt = data + ndim * obs;
            mindist[obs] = std::numeric_limits<double>::infinity();
            double to_add = std::numeric_limits<double>::infinity();

            for (CLUSTER_t c2 = 0; c2 < ncenters; ++c2) {
                const DATA_t* target = centers + ndim * c2;
                double dist = 0;
                for (int d = 0; d < ndim; ++d) {
                    double delta = pt[d] - target[d];
                    dist += delta * delta;
                }

                // We need to compute the WCSS as if the existing cluster
                // was still there, but we only want to store the minimum 
                // distance after removing the existing cluster.
                if (dist < to_add) {
                    to_add = dist;
                }
                if (c2 != c) {
                    if (dist < mindist[obs]) {
                        clusters[obs] = c2;
                        mindist[obs] = dist;
                    }
                }
            } 

            wcss += to_add;
        }

        cumulative[0] = mindist[0];
        for (INDEX_t i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }

        INDEX_t chosen_id = 0;
        bool found = false;
        for (int i = 0; i < kmeans::Reinitialize<>::Defaults::iterations; ++i) {
            auto id = kmeans::weighted_sample(cumulative, mindist, nobs, eng);
            const DATA_t* target = data + id * ndim;
            auto mindist_copy = mindist;

            for (INDEX_t obs = 0; obs < nobs; ++obs) {
                const DATA_t* pt = data + ndim * obs;
                double dist = 0;
                for (int d = 0; d < ndim; ++d) {
                    double delta = pt[d] - target[d];
                    dist += delta * delta;
                }
                if (dist < mindist_copy[obs]) {
                    mindist_copy[obs] = dist;
                }
            } 

            double wcss2 = std::accumulate(mindist_copy.begin(), mindist_copy.end(), 0.0);
            if (wcss2 < wcss) {
                found = true;
                chosen_id = id;
                mindist.swap(mindist_copy);
                break;
            }
        }

        if (found) {
            auto chosen_ptr = data + chosen_id * ndim;
            auto target_ptr = centers + c * ndim;
            std::copy(chosen_ptr, chosen_ptr + ndim, target_ptr);
        }
    }

    return;
}

using ReinitializeTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(ReinitializeTest, CompareToRef) {
    auto param = GetParam();
    assemble(param);
    int ncenters = std::get<2>(param);
    auto centers = create_centers(data.data(), ncenters);
    size_t seed = nc * nr / ncenters;

    auto copy = centers;
    std::vector<int> clusters(nc);
    kmeans::Reinitialize runner;
    runner.set_seed(seed);
    runner.run(nr, nc, data.data(), ncenters, copy.data(), clusters.data());

    auto ref = centers;
    std::mt19937_64 eng2(seed);
    ref_reinitialize_centers(nr, nc, data.data(), ncenters, ref.data(), eng2);

    EXPECT_EQ(copy, ref);

    // Checking that the clusters are actually the best.
    kmeans::QuickSearch searcher(nr, ncenters, copy.data());
    for (int c = 0; c < nc; ++c) {
        auto best = searcher.find(data.data() + c * nr);
        EXPECT_EQ(best, clusters[c]);
    }
}

TEST_P(ReinitializeTest, PreClustered) {
    auto param = GetParam();
    assemble(param);
    int ncenters = std::get<2>(param);
    auto centers = create_centers(data.data(), ncenters);
    size_t seed = nc * nr / ncenters;

    std::vector<int> clusters(nc);
    kmeans::QuickSearch searcher(nr, ncenters, centers.data());
    for (int c = 0; c < nc; ++c) {
        clusters[c] = searcher.find(data.data() + c * nr);
    }

    auto center_copy = centers;
    auto cluster_copy = clusters;
    kmeans::Reinitialize runner;
    runner.set_seed(seed);
    runner.set_recompute_clusters(true).run(nr, nc, data.data(), ncenters, center_copy.data(), cluster_copy.data());

    auto center_copy2 = centers;
    auto cluster_copy2 = clusters;
    runner.set_recompute_clusters(false).run(nr, nc, data.data(), ncenters, center_copy2.data(), cluster_copy2.data());

    EXPECT_EQ(center_copy, center_copy2);
    EXPECT_EQ(cluster_copy, cluster_copy2);
}

INSTANTIATE_TEST_CASE_P(
    Reinitialize,
    ReinitializeTest,
    ::testing::Combine(
        ::testing::Values(5), // number of dimensions
        ::testing::Values(57, 342), // number of observations 
        ::testing::Values(5, 10, 20) // number of centers 
    )
);
