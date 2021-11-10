#include <gtest/gtest.h>
#include "kmeans/reinitialize.hpp"
#include "TestCore.h"

#include <vector>
#include <limits>

template<typename DATA_t = double, typename INDEX_t = int, typename CLUSTER_t = int, class ENGINE>
void ref_reinitialize_centers(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, ENGINE& eng) {
    std::vector<DATA_t> mindist(nobs);
    std::vector<DATA_t> cumulative(nobs);

    for (CLUSTER_t c = 0; c < ncenters; ++c) {
        // Finding the new closest cluster for each point.
        for (INDEX_t obs = 0; obs < nobs; ++obs) {
            const DATA_t* pt = data + ndim * obs;
            mindist[obs] = std::numeric_limits<double>::infinity();

            for (CLUSTER_t c2 = 0; c2 < ncenters; ++c2) {
                if (c2 == c) {
                    continue;
                }

                DATA_t* target = centers + ndim * c2;
                double dist = 0;
                for (int d = 0; d < ndim; ++d) {
                    dist += (pt[d] - target[d]) * (pt[d] - target[d]);
                }

                if (dist < mindist[obs]) {
                    mindist[obs] = dist;
                }
            } 
        }

        cumulative[0] = mindist[0];
        for (INDEX_t i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }
        auto chosen_id = kmeans::weighted_sample(cumulative, mindist, nobs, eng);

        auto chosen_ptr = data + chosen_id * ndim;
        auto target_ptr = centers + c * ndim;
        std::copy(chosen_ptr, chosen_ptr + ndim, target_ptr);
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
    std::mt19937_64 eng1(seed);
    kmeans::reinitialize_centers(nr, nc, data.data(), ncenters, copy.data(), clusters.data(), eng1, true);

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
    std::mt19937_64 eng1(seed);
    kmeans::reinitialize_centers(nr, nc, data.data(), ncenters, center_copy.data(), cluster_copy.data(), eng1, true);

    auto center_copy2 = centers;
    auto cluster_copy2 = clusters;
    std::mt19937_64 eng2(seed);
    kmeans::reinitialize_centers(nr, nc, data.data(), ncenters, center_copy2.data(), cluster_copy2.data(), eng2, false);

    EXPECT_EQ(center_copy, center_copy2);
    EXPECT_EQ(cluster_copy, cluster_copy2);
}

INSTANTIATE_TEST_CASE_P(
    Reinitialize,
    ReinitializeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(57, 345), // number of observations 
        ::testing::Values(5, 10, 20) // number of centers 
    )
);
