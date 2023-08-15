#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/MiniBatch.hpp"

using MiniBatchBasicTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(MiniBatchBasicTest, Sweep) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    auto centers = create_centers(data.data(), ncenters);
    std::vector<int> clusters(nc);
    auto mb = kmeans::MiniBatch<>().run(nr, nc, data.data(), ncenters, centers.data(), clusters.data());

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, mb.sizes);
    for (auto c : counts) {
        EXPECT_TRUE(c > 0); 
    }

    EXPECT_TRUE(mb.iterations > 0);

    // Checking that the WCSS calculations are correct.
    const auto& wcss = mb.withinss;
    for (int i = 0; i < ncenters; ++i) {
        if (counts[i] > 1) {
            EXPECT_TRUE(wcss[i] > 0);
        } else {
            EXPECT_EQ(wcss[i], 0);
        }
    }

    // Checking that parallelization gives the same result.
    {
        auto pcenters = create_centers(data.data(), ncenters);
        std::vector<int> pclusters(nc);
        auto pmb = kmeans::MiniBatch<>().set_num_threads(3).run(nr, nc, data.data(), ncenters, pcenters.data(), pclusters.data());
        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
        EXPECT_EQ(pmb.withinss, wcss);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MiniBatch,
    MiniBatchBasicTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

using MiniBatchConstantTest = TestParamCore<std::tuple<int, int> >;

TEST_P(MiniBatchConstantTest, TooMany) {
    auto param = GetParam();
    assemble(param);

    std::vector<double> centers(data.size());
    std::vector<int> clusters(nc);

    // In this case, the centroids and points should be equal.
    // We won't bother applying more checks as these are redundant
    // with the tests in HartiganWong.cpp covering the same code.
    auto mb = kmeans::MiniBatch<>().run(nr, nc, data.data(), nc, centers.data(), clusters.data());

    EXPECT_EQ(data, centers);
    EXPECT_EQ(mb.withinss, std::vector<double>(nc));
    EXPECT_EQ(mb.sizes, std::vector<int>(nc, 1));
    EXPECT_EQ(mb.iterations, 0);

    std::vector<int> ref(nc);
    std::iota(ref.begin(), ref.end(), 0);
    EXPECT_EQ(ref, clusters);
}

INSTANTIATE_TEST_SUITE_P(
    MiniBatch,
    MiniBatchConstantTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000) // number of observations 
    )
);
