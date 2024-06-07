#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineLloyd.hpp"

class RefineLloydBasicTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(RefineLloydBasicTest, Sweep) {
    auto ncenters = std::get<1>(GetParam());

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto centers = create_centers(ncenters);
    auto original = centers;

    std::vector<int> clusters(nc);
    kmeans::RefineLloyd ll;
    auto res = ll.run(mat, ncenters, centers.data(), clusters.data());

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.sizes);
    EXPECT_TRUE(res.iterations > 0);

    // Checking that parallelization gives the same result.
    {
        kmeans::RefineLloydOptions popt;
        popt.num_threads = 3;
        kmeans::RefineLloyd pll(popt);

        auto pcenters = original;
        std::vector<int> pclusters(nc);
        pll.run(mat, ncenters, pcenters.data(), pclusters.data());

        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
    }
}

TEST_P(RefineLloydBasicTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_jittered_matrix(ncenters);
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    // Lloyd should give us back the perfect clusters.
    std::vector<int> clusters(nc);
    kmeans::RefineLloyd ll;
    auto res = ll.run(mat, ncenters, dups.centers.data(), clusters.data());

    EXPECT_EQ(clusters, dups.clusters);
}

INSTANTIATE_TEST_SUITE_P(
    RefineLloyd,
    RefineLloydBasicTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations 
        ),
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

class RefineLloydConstantTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 20, 50 });
    }
};

TEST_F(RefineLloydConstantTest, Extremes) {
    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineLloyd ll;

    {
        std::vector<double> centers(nr * nc);
        std::vector<int> clusters(nc);
        auto res = ll.run(mat, nc, centers.data(), clusters.data());
        EXPECT_EQ(data, centers);
    }

    {
        auto res0 = ll.run(mat, 0, NULL, NULL);
        EXPECT_TRUE(res0.sizes.empty());
    }
}
