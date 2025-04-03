#include "TestCore.h"

#include <memory>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/kmeans.hpp"

class KmeansBasicTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(KmeansBasicTest, Sweep) {
    kmeans::SimpleMatrix<int, double> mat(nr, nc, data.data());
    kmeans::InitializeRandom<int, double, int, double> init_rand;
    kmeans::RefineHartiganWong<int, double, int, double> ref_hw;

    auto ncenters = std::get<1>(GetParam());
    auto res = kmeans::compute(mat, init_rand, ref_hw, ncenters);
    EXPECT_EQ(res.clusters.size(), nc);
    EXPECT_EQ(res.centers.size(), ncenters * nr);

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    for (auto c : res.clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.details.sizes);
    EXPECT_TRUE(res.details.iterations > 0);

    // Get some coverage on the other overload.
    std::vector<int> clusters(nc);
    std::vector<double> centroids(nr * ncenters);
    auto deets = kmeans::compute(mat, init_rand, ref_hw, ncenters, centroids.data(), clusters.data());
    EXPECT_EQ(clusters, res.clusters);
    EXPECT_EQ(centroids, res.centers);
}

INSTANTIATE_TEST_SUITE_P(
    Kmeans,
    KmeansBasicTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations 
        ),
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

class KmeansExpandedTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 50, 10 }); 
    }
};

TEST_F(KmeansExpandedTest, Basic) {
    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto res = kmeans::compute(mat, kmeans::InitializeRandom<int, double, int, double>(), kmeans::RefineHartiganWong<int, double, int, double>(), nc + 10);
    EXPECT_EQ(res.centers.size(), nr * (nc + 10));
    EXPECT_EQ(res.clusters.size(), nc);

    // Sizes are correctly resized.
    std::vector<int> counts(nc, 1);
    counts.insert(counts.end(), 10, 0);
    EXPECT_EQ(counts, res.details.sizes);
}

class KmeansSanityTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(KmeansSanityTest, SanityCheck) {
    auto param = GetParam();
    auto ncenters = std::get<1>(param);

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
    std::unique_ptr<kmeans::Refine<int, double, int, double> > ptr;
    auto algo = std::get<2>(param);
    if (algo == 1) {
        auto xptr = new kmeans::RefineHartiganWong<int, double, int, double>();
        ptr.reset(xptr);
    } else if (algo == 2) {
        auto xptr = new kmeans::RefineLloyd<int, double, int, double>();
        ptr.reset(xptr);
    } else if (algo == 3) {
        auto xptr = new kmeans::RefineMiniBatch<int, double, int, double>();
        ptr.reset(xptr);
    }

    auto res = kmeans::compute(kmeans::SimpleMatrix<int, double>(nr, nc, data.data()), kmeans::InitializeKmeanspp<int, double, int, double>(), *ptr, ncenters);

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
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations 
        ),
        ::testing::Values(2, 5, 10), // number of clusters 
        ::testing::Values(1, 2, 3) // algorithm
    )
);

