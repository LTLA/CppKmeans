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

using RefineHartiganWongConstantTest = TestCore<::testing::Test>;

TEST_F(RefineHartiganWongConstantTest, Extremes) {
    nr = 20;
    nc = 50;
    assemble();

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineHartiganWong ll;

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
