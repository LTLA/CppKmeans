#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineMiniBatch.hpp"

class RefineMiniBatchBasicTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(RefineMiniBatchBasicTest, Sweep) {
    auto ncenters = std::get<1>(GetParam());

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto centers = create_centers(ncenters);
    auto original = centers;
    std::vector<int> clusters(nc);

    kmeans::RefineMiniBatchOptions opt;
    opt.batch_size = 100; // reducing the batch size so that the sampling actually does something.
    kmeans::RefineMiniBatch mb(opt);
    auto res = mb.run(mat, ncenters, centers.data(), clusters.data());

    // Checking that there's the specified number of clusters. 
    std::vector<int> counts(ncenters);
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.sizes);
    EXPECT_TRUE(res.iterations > 0);

    // Checking that parallelization gives the same result.
    {
        auto pcenters = original;
        std::vector<int> pclusters(nc);

        auto popt = opt;
        popt.num_threads = 3;
        kmeans::RefineMiniBatch pmb(popt);
        pmb.run(mat, ncenters, pcenters.data(), pclusters.data());

        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
    }
}

TEST_P(RefineMiniBatchBasicTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_jittered_matrix(ncenters);
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    // Should give us back the perfect clusters.
    std::vector<int> clusters(nc);
    kmeans::RefineMiniBatch mb;
    auto res = mb.run(mat, ncenters, dups.centers.data(), clusters.data());

    EXPECT_EQ(clusters, dups.clusters);
}

INSTANTIATE_TEST_SUITE_P(
    RefineMiniBatch,
    RefineMiniBatchBasicTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations 
        ),
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

class RefineMiniBatchConstantTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 20, 50 });
    }
};

TEST_F(RefineMiniBatchConstantTest, Extremes) {
    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineMiniBatch mb;

    {
        std::vector<double> centers(nr * nc);
        std::vector<int> clusters(nc);
        auto res = mb.run(mat, nc, centers.data(), clusters.data());
        EXPECT_EQ(data, centers);
    }

    {
        auto res0 = mb.run(mat, 0, NULL, NULL);
        EXPECT_TRUE(res0.sizes.empty());
    }
}
