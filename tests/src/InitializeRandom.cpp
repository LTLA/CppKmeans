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
    auto output = kmeans::sample_without_replacement(nc, ncenters, rng);
    auto copy = output;

    EXPECT_EQ(output.size(), ncenters);
    int last = -1;
    std::sort(output.begin(), output.end());
    for (auto o : output) {
        EXPECT_TRUE(o > last); // no duplicates
        EXPECT_TRUE(o < nc); // in range
        last = o;
    }

    // Checks that the class does the right thing.
    kmeans::InitializeRandom init;
    init.set_seed(ncenters * 10);
    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(nr, nc, data.data(), ncenters, centers.data(), NULL);
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

TEST_P(RandomInitializationTest, Deterministic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    std::mt19937_64 rng(ncenters * 10);
    auto copy = kmeans::sample_without_replacement(nc, ncenters, rng);

    // Consistent results with the same initialization.
    std::mt19937_64 rng2(ncenters * 10);
    auto output2 = kmeans::sample_without_replacement(nc, ncenters, rng2);
    EXPECT_EQ(copy, output2);

    // Different results with a different seed (only works if num obs is reasonably larger than num centers).
    std::mt19937_64 rng3(ncenters * 11);
    auto output3 = kmeans::sample_without_replacement(nc, ncenters, rng3);
    EXPECT_NE(copy, output3);
}

INSTANTIATE_TEST_CASE_P(
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
    std::mt19937_64 rng(nc* 10);
   
    std::vector<int> expected(nc);
    std::iota(expected.begin(), expected.end(), 0);

    auto soutput = kmeans::sample_without_replacement(nc, nc, rng);
    EXPECT_EQ(soutput, expected);

    auto soutput2 = kmeans::sample_without_replacement(nc, nc + 1, rng);
    EXPECT_EQ(soutput2, expected);
}

INSTANTIATE_TEST_CASE_P(
    RandomInitialization,
    RandomInitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50) // number of dimensions
    )
);
