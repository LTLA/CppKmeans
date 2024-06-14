#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineHartiganWong.hpp"

class RefineHartiganWongBasicTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(RefineHartiganWongBasicTest, Sweep) {
    auto ncenters = std::get<1>(GetParam());

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto centers = create_centers(ncenters);
    auto original = centers;
    std::vector<int> clusters(nc);

    kmeans::RefineHartiganWong hw;
    auto res = hw.run(mat, ncenters, centers.data(), clusters.data());

    // Checking that there's the specified number of clusters. 
    std::vector<int> counts(ncenters);
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, res.sizes);
    EXPECT_TRUE(res.iterations > 0);

    // Checking paralleization yields the same results.
    {
        kmeans::RefineHartiganWongOptions popt;
        popt.num_threads = 3;
        kmeans::RefineHartiganWong phw(popt);

        auto pcenters = original;
        std::vector<int> pclusters(nc);

        auto pres = phw.run(mat, ncenters, pcenters.data(), pclusters.data());
        EXPECT_EQ(pcenters, centers);
        EXPECT_EQ(pclusters, clusters);
    }
}

TEST_P(RefineHartiganWongBasicTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_jittered_matrix(ncenters);
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    // HartiganWong should give us back the perfect clusters.
    std::vector<int> clusters(nc);
    kmeans::RefineHartiganWong hw;
    auto res = hw.run(mat, ncenters, dups.centers.data(), clusters.data());

    EXPECT_EQ(clusters, dups.clusters);
}

INSTANTIATE_TEST_SUITE_P(
    RefineHartiganWong,
    RefineHartiganWongBasicTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations 
        ),
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

class RefineHartiganWongConstantTest : public TestCore, public ::testing::Test { 
protected:
    void SetUp() {
        assemble({ 20, 50 });
    }
};

TEST_F(RefineHartiganWongConstantTest, Extremes) {
    kmeans::SimpleMatrix mat(nr, nc, data.data());
    kmeans::RefineHartiganWong hw;

    {
        std::vector<double> centers(nr * nc);
        std::vector<int> clusters(nc);
        auto res = hw.run(mat, nc, centers.data(), clusters.data());
        EXPECT_EQ(data, centers);
    }

    {
        auto res0 = hw.run(mat, 0, NULL, NULL);
        EXPECT_TRUE(res0.sizes.empty());
    }
}

TEST(RefineHartiganWong, Options) {
    kmeans::RefineHartiganWongOptions opt;
    opt.num_threads = 10;
    kmeans::RefineHartiganWong ref(opt);
    EXPECT_EQ(ref.get_options().num_threads, 10);

    ref.get_options().num_threads = 9;
    EXPECT_EQ(ref.get_options().num_threads, 9);
}
