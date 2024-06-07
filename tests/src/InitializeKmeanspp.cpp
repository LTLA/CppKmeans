#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeKmeanspp.hpp"

class KmeansppInitializationTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(KmeansppInitializationTest, Internals) {
    auto ncenters = std::get<1>(GetParam());

    kmeans::SimpleMatrix mat(nr, nc, data.data());
    int seed = ncenters * 10 + nr + nc;
    auto output = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed, 1);
    EXPECT_EQ(output.size(), ncenters);

    // Check that a reasonable selection is made.
    {
        auto copy = output;
        int last = -1;
        std::sort(copy.begin(), copy.end());
        for (auto o : copy) {
            EXPECT_TRUE(o > last); // no duplicates
            EXPECT_TRUE(o < nc); // in range
            last = o;
        }
    }

    // Consistent results with the same initialization.
    {
        auto output2 = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed, 1);
        EXPECT_EQ(output, output2);
        
        // Different results with a different seed (note that this only works
        // if num obs is reasonably larger than num centers).
        auto output3 = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed + 1, 1);
        EXPECT_NE(output, output3);
    }

    // Check that parallelization gives the same result.
    {
        auto output2 = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed, 3);
        EXPECT_EQ(output, output2);
    }
}

TEST_P(KmeansppInitializationTest, Basic) {
    auto ncenters = std::get<1>(GetParam());
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    kmeans::InitializeKmeanspp init;
    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(mat, ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);

    auto matched = match_to_data(ncenters, centers);
    for (auto m : matched) {
        EXPECT_TRUE(m >= 0);
        EXPECT_TRUE(m < nc);
    }

    std::sort(matched.begin(), matched.end());
    for (size_t i = 1; i < matched.size(); ++i) {
        EXPECT_TRUE(matched[i] > matched[i-1]);
    }

    // Same results with parallelization.
    {
        kmeans::InitializeKmeansppOptions popt;
        popt.num_threads = 3;
        kmeans::InitializeKmeanspp pinit(popt);
        std::vector<double> pcenters(nr * ncenters);
        init.run(mat, ncenters, pcenters.data());
        EXPECT_EQ(pcenters, centers);
    }
}

TEST_P(KmeansppInitializationTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_duplicate_matrix(ncenters);

    // Expect one entry from each of the first 'ncenters' elements;
    // all others are duplicates and should have sampling probabilities of zero.
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());
    auto seed = ncenters * 100;
    auto output = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed, 1);

    EXPECT_EQ(output.size(), ncenters);
    for (auto& o : output) {
        o = dups.clusters[o];
    }
    std::sort(output.begin(), output.end());

    std::vector<int> expected(ncenters);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(expected, output);

    // If more clusters are requested, we detect that only duplicates are available and we bail early.
    auto output2 = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters + 1, seed, 1);
    EXPECT_EQ(output2.size(), ncenters);
    for (auto& o : output2) {
        o = dups.clusters[o];
    }
    std::sort(output2.begin(), output2.end());
    EXPECT_EQ(expected, output2);
}

INSTANTIATE_TEST_SUITE_P(
    KmeansppInitialization,
    KmeansppInitializationTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations
        ),
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

class KmeansppInitializationEdgeTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 10, 20 });
    }
};

TEST_F(KmeansppInitializationEdgeTest, TooManyClusters) {
    kmeans::InitializeKmeansppOptions opt;
    opt.seed = nc * 10;
    kmeans::InitializeKmeanspp init(opt);

    std::vector<double> centers(nc * nr);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc, centers.data());
    EXPECT_EQ(nfilled, nc);

    // Check that there's one representative from each cluster.
    auto matched = match_to_data(nc, centers);
    std::sort(matched.begin(), matched.end());
    std::vector<int> expected(nc);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(matched, expected);

    // Same as if we have more clusters.
    std::vector<double> centers2(nc * nr);
    auto nfilled2 = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc + 10, centers2.data());
    EXPECT_EQ(nfilled2, nc);
    EXPECT_EQ(centers2, centers);
}
