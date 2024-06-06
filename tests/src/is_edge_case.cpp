#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/is_edge_case.hpp"
#include "kmeans/SimpleMatrix.hpp"

using EdgeCaseTest = TestParamCore<std::tuple<int, int> >;

TEST_P(EdgeCaseTest, TooMany) {
    auto param = GetParam();
    assemble(param);
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    {
        EXPECT_TRUE(kmeans::internal::is_edge_case(mat.num_observations(), nc));

        std::vector<double> centers(data.size());
        std::vector<int> clusters(nc);
        auto res = kmeans::internal::process_edge_case(mat, nc, centers.data(), clusters.data());

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
        EXPECT_TRUE(kmeans::internal::is_edge_case(mat.num_observations(), nc));

        std::vector<double> centers(data.size() + nr);
        std::vector<int> clusters(nc);

        auto res = kmeans::internal::process_edge_case(mat, nc + 1, centers.data(), clusters.data());
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

TEST_P(EdgeCaseTest, TooFew) {
    auto param = GetParam();
    assemble(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    EXPECT_TRUE(kmeans::internal::is_edge_case(mat.num_observations(), 1));

    std::vector<double> centers(nr);
    std::vector<int> clusters(nc);
    auto res = kmeans::internal::process_edge_case(mat, 1, centers.data(), clusters.data());

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
    EXPECT_TRUE(kmeans::internal::is_edge_case(mat.num_observations(), 0));
    auto res0 = kmeans::internal::process_edge_case(mat, 0, centers.data(), clusters.data());
    EXPECT_TRUE(res0.sizes.empty());
}

INSTANTIATE_TEST_SUITE_P(
    EdgeCase,
    EdgeCaseTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200) // number of observations 
    )
);
