#include "TestCore.h"

#include <memory>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/kmeans.hpp"

using KmeansBasicTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(KmeansBasicTest, Sweep) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto res = kmeans::compute(mat, kmeans::InitializeRandom(), kmeans::RefineHartiganWong(), ncenters);
    EXPECT_EQ(res.clusters.size(), nc);
    EXPECT_EQ(res.centers.size(), ncenters * nr);

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    for (auto c : res.clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.details.sizes);
    for (auto c : counts) {
        EXPECT_TRUE(c > 0); 
    }

    EXPECT_TRUE(res.details.iterations > 0);
}

INSTANTIATE_TEST_SUITE_P(
    Kmeans,
    KmeansBasicTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

using KmeansExpandedTest = TestCore<::testing::Test>;

TEST_F(KmeansExpandedTest, Basic) {
    nr = 50;
    nc = 10;
    assemble();

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto res = kmeans::compute(mat, kmeans::InitializeRandom(), kmeans::RefineHartiganWong(), nc + 10);
    EXPECT_EQ(res.centers.size(), nr * (nc + 10));
    EXPECT_EQ(res.clusters.size(), nc);

    // Sizes are correctly resized.
    std::vector<int> counts(nc, 1);
    counts.insert(counts.end(), 10, 0);
    EXPECT_EQ(counts, res.details.sizes);
}

using KmeansSanityTest = TestParamCore<std::tuple<int, int, int, int> >;

TEST_P(KmeansSanityTest, SanityCheck) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    // Adding known structure by scaling down the variance and adding a predictable offset per cluster.
    auto dIt = data.begin();
    for (int c = 0; c < nc; ++c) {
        double additive = c % ncenters;
        for (int r = 0; r < nr; ++r, ++dIt) {
            (*dIt) /= 1000;
            (*dIt) += additive;
        }
    }

    // Switching between algorithms.
    std::unique_ptr<kmeans::Refine<> > ptr;
    auto algo = std::get<3>(param);
    if (algo == 1) {
        auto xptr = new kmeans::RefineHartiganWong<>();
        ptr.reset(xptr);
    } else if (algo == 2) {
        auto xptr = new kmeans::RefineLloyd<>();
        ptr.reset(xptr);
    } else if (algo == 3) {
        auto xptr = new kmeans::RefineMiniBatch<>();
        ptr.reset(xptr);
    }

    auto res = kmeans::compute(kmeans::SimpleMatrix(nr, nc, data.data()), kmeans::InitializeKmeanspp(), *ptr, ncenters);

    // Checking that every 'ncenters'-th element is the same.
    std::vector<int> last_known(ncenters, -1);
    for (int c = 0; c < nc; ++c) {
        int& x = last_known[c % ncenters];
        if (x < 0) {
            x = res.clusters[c];
        } else {
            EXPECT_EQ(x, res.clusters[c]);
        }
    }

    // Checking that there are, in fact, 'ncenters' unique cluster IDs.
    std::sort(last_known.begin(), last_known.end());
    EXPECT_TRUE(last_known[0] >= 0);
    for (int i = 1; i < ncenters; ++i) {
        EXPECT_TRUE(last_known[i] >= 0);
        EXPECT_NE(last_known[i], last_known[i-1]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Kmeans,
    KmeansSanityTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10), // number of clusters 
        ::testing::Values(1, 2, 3) // algorithm
    )
);

