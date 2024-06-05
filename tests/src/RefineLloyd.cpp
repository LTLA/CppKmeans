#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineLloyd.hpp"

using RefineLloydBasicTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(RefineLloydBasicTest, Sweep) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto centers = create_centers(data.data(), ncenters);
    std::vector<int> clusters(nc);

    kmeans::RefineLloydOptions opt;
    kmeans::RefineLloyd ll(opt);
    auto res = ll.run(mat, ncenters, centers.data(), clusters.data());

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.sizes);
    for (auto c : counts) {
        EXPECT_TRUE(c > 0); 
    }

    EXPECT_TRUE(res.iterations > 0);

    // Checking that parallelization gives the same result.
    {
        auto pcenters = create_centers(data.data(), ncenters);
        std::vector<int> pclusters(nc);

        kmeans::RefineLloydOptions popt;
        opt.num_threads = 3;
        kmeans::RefineLloyd pll(popt);

        pll.run(mat, ncenters, pcenters.data(), pclusters.data());
        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
    }
}

INSTANTIATE_TEST_SUITE_P(
    RefineLloyd,
    RefineLloydBasicTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

using RefineLloydConstantTest = TestParamCore<std::tuple<int, int> >;

TEST_P(RefineLloydConstantTest, TooMany) {
    auto param = GetParam();
    assemble(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineLloydOptions opt;
    kmeans::RefineLloyd ll(opt);

    {
        std::vector<double> centers(data.size());
        std::vector<int> clusters(nc);
        auto res = ll.run(mat, nc, centers.data(), clusters.data());

        // Checking that the averages are just equal to the points.
        EXPECT_EQ(data, centers);
        EXPECT_EQ(res.sizes, std::vector<int>(nc, 1));
        EXPECT_EQ(res.iterations, 0);
        EXPECT_EQ(res.status, 0);

        std::vector<int> ref(nc);
        std::iota(ref.begin(), ref.end(), 0);
        EXPECT_EQ(ref, clusters);
    }

    {
        std::vector<double> centers(data.size() + nr);
        std::vector<int> clusters(nc);

        auto res = ll.run(mat, nc + 1, centers.data(), clusters.data());
        EXPECT_EQ(res.status, 0);

        std::vector<int> ref(nc);
        std::iota(ref.begin(), ref.end(), 0);
        EXPECT_EQ(ref, clusters);

        std::vector<double> truncated(centers.begin(), centers.begin() + data.size());
        EXPECT_EQ(data, truncated);

        std::vector<int> expected_sizes(nc, 1);
        expected_sizes.push_back(0);
        EXPECT_EQ(res.sizes, expected_sizes);
    }
}

TEST_P(RefineLloydConstantTest, TooFew) {
    auto param = GetParam();
    assemble(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineLloydOptions opt;
    kmeans::RefineLloyd ll(opt);

    std::vector<double> centers(nr);
    std::vector<int> clusters(nc);
    auto res = ll.run(mat, 1, centers.data(), clusters.data());

    std::vector<double> averages(nr);
    size_t i = 0;
    for (auto d : data) {
        averages[i] += d;
        ++i;
        i %= nr;
    }
    for (auto& a : averages) {
        a /= nc;
    }

    EXPECT_EQ(centers, averages);
    EXPECT_EQ(res.iterations, 0);

    // no points at all.
    auto res0 = ll.run(mat, 0, NULL, NULL);
    EXPECT_TRUE(res0.sizes.empty());
}

INSTANTIATE_TEST_SUITE_P(
    RefineLloyd,
    RefineLloydConstantTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000) // number of observations 
    )
);
