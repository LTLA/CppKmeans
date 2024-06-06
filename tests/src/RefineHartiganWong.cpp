#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineHartiganWong.hpp"

using RefineHartiganWongBasicTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(RefineHartiganWongBasicTest, Sweep) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto centers = create_centers(data.data(), ncenters);
    std::vector<int> clusters(nc);

    kmeans::RefineHartiganWong hw;
    auto res = hw.run(mat, ncenters, centers.data(), clusters.data());

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

    // Checking paralleization yields the same results.
    {
        kmeans::RefineHartiganWongOptions popt;
        popt.num_threads = 3;
        kmeans::RefineHartiganWong phw(popt);

        auto pcenters = create_centers(data.data(), ncenters);
        std::vector<int> pclusters(nc);

        auto pres = phw.run(mat, ncenters, pcenters.data(), pclusters.data());
        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
    }
}

TEST_P(RefineHartiganWongBasicTest, Sanity) {
    auto param = GetParam();
    assemble(param);

    // Duplicating the first 'ncenters' elements over and over again.
    auto ncenters = std::get<2>(param);
    auto dups = create_duplicate_matrix(ncenters);
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    // HartiganWong should give us back the perfect clusters.
    std::vector<int> clusters(nc);
    kmeans::RefineHartiganWong hw;
    auto res = hw.run(mat, ncenters, dups.centers.data(), clusters.data());

    EXPECT_EQ(clusters, dups.chosen);
}

INSTANTIATE_TEST_SUITE_P(
    RefineHartiganWong,
    RefineHartiganWongBasicTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

using RefineHartiganWongConstantTest = TestParamCore<std::tuple<int, int> >;

TEST_P(RefineHartiganWongConstantTest, TooMany) {
    auto param = GetParam();
    assemble(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineHartiganWong hw;

    {
        std::vector<double> centers(data.size());
        std::vector<int> clusters(nc);
        auto res = hw.run(mat, nc, centers.data(), clusters.data());

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

        auto res = hw.run(mat, nc + 1, centers.data(), clusters.data());
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

TEST_P(RefineHartiganWongConstantTest, TooFew) {
    auto param = GetParam();
    assemble(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineHartiganWong hw;

    std::vector<double> centers(nr);
    std::vector<int> clusters(nc);
    auto res = hw.run(mat, 1, centers.data(), clusters.data());

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
    auto res0 = hw.run(mat, 0, NULL, NULL);
    EXPECT_TRUE(res0.sizes.empty());
}

INSTANTIATE_TEST_SUITE_P(
    RefineHartiganWong,
    RefineHartiganWongConstantTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200) // number of observations 
    )
);
