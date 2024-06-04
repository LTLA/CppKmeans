#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeRandom.hpp"

using RandomInitializationTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(RandomInitializationTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    std::mt19937_64 rng(ncenters * 10);
    std::vector<int> output(ncenters);
    aarand::sample(nc, ncenters, output.data(), rng);
    auto copy = output;

    // Double-check that the sampling is done correctly.
    EXPECT_EQ(output.size(), ncenters);
    int last = -1;
    std::sort(output.begin(), output.end());
    for (auto o : output) {
        EXPECT_TRUE(o > last); // no duplicates
        EXPECT_TRUE(o < nc); // in range
        last = o;
    }

    // Checks that the class does the right thing.
    kmeans::InitializeRandomOptions opt;
    opt.seed = ncenters * 10;
    kmeans::InitializeRandom init(opt);
    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);

    auto cptr = centers.data();
    for (auto c : copy) {
        std::vector<double> obs(cptr, cptr + nr);
        cptr += nr;
        auto optr = data.data() + c * nr;
        std::vector<double> exp(optr, optr + nr);
        EXPECT_EQ(obs, exp);
    }
}

INSTANTIATE_TEST_SUITE_P(
    RandomInitialization,
    RandomInitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using RandomInitializationEdgeTest = TestParamCore<std::tuple<int, int> >;

TEST_P(RandomInitializationEdgeTest, TooManyClusters) {
    auto param = GetParam();
    assemble(param);

    kmeans::InitializeRandomOptions opt;
    opt.seed = nc * 10;
    kmeans::InitializeRandom init(opt);

    std::vector<double> centers(nc * nr);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc, centers.data());
    EXPECT_EQ(nfilled, nc);
    EXPECT_EQ(centers, data);

    std::fill(centers.begin(), centers.end(), 0);
    nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc + 10, centers.data());
    EXPECT_EQ(nfilled, nc);
    EXPECT_EQ(centers, data);
}

INSTANTIATE_TEST_SUITE_P(
    RandomInitialization,
    RandomInitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50)  // number of observations 
    )
);
