#include "kmeans/initialization.hpp"
#include <random>
#include <vector>
#include "TestCore.h"

using InitializationTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(InitializationTest, Simple) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    std::mt19937_64 rng(ncenters * 10);
    auto output = kmeans::simple_initialization(nc, ncenters, rng);
    auto copy = output;

    EXPECT_EQ(output.size(), ncenters);
    int last = -1;
    std::sort(output.begin(), output.end());
    for (auto o : output) {
        EXPECT_TRUE(o > last); // no duplicates
        EXPECT_TRUE(o < nc); // in range
        last = o;
    }

    // Consistent results with the same initialization.
    std::mt19937_64 rng2(ncenters * 10);
    auto output2 = kmeans::simple_initialization(nc, ncenters, rng2);
    EXPECT_EQ(copy, output2);

    // Different results with a different seed (only works if num obs is reasonably larger than num centers).
    std::mt19937_64 rng3(ncenters * 11);
    auto output3 = kmeans::simple_initialization(nc, ncenters, rng3);
    EXPECT_NE(copy, output3);
}

TEST_P(InitializationTest, Weighted) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    std::mt19937_64 rng(ncenters * 10);
    auto output = kmeans::weighted_initialization(nr, nc, data.data(), ncenters, rng);
    auto copy = output;

    EXPECT_EQ(output.size(), ncenters);
    int last = -1;
    std::sort(output.begin(), output.end());
    for (auto o : output) {
        EXPECT_TRUE(o > last); // no duplicates
        EXPECT_TRUE(o < nc); // in range
        last = o;
    }

    // Consistent results with the same initialization.
    std::mt19937_64 rng2(ncenters * 10);
    auto output2 = kmeans::weighted_initialization(nr, nc, data.data(), ncenters, rng2);
    EXPECT_EQ(copy, output2);
    
    // Different results with a different seed (only works if num obs is reasonably larger than num centers).
    std::mt19937_64 rng3(ncenters * 11);
    auto output3 = kmeans::weighted_initialization(nr, nc, data.data(), ncenters, rng3);
    EXPECT_NE(copy, output3);
}

INSTANTIATE_TEST_CASE_P(
    Initialization,
    InitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using WeightedInitializationTest = TestParamCore<std::tuple<int, int, int> >;

std::vector<int> one_to_n (int n) {
    std::vector<int> output(n);
    std::iota(output.begin(), output.end(), 0);
    return output;
}

TEST_P(WeightedInitializationTest, Sanity) {
    auto param = GetParam();
    assemble(param);

    // Duplicating the first 'nc' elements over and over again.
    int ncenters = std::get<2>(param);

    std::vector<int> choices(nc);
    std::iota(choices.begin(), choices.begin() + ncenters, 0);

    std::mt19937_64 rng(nc * 10);
    auto dIt = data.begin() + ncenters * nr;
    for (int c = ncenters; c < nc; ++c, dIt += nr) {
        auto chosen = rng() % ncenters;
        auto cIt = data.begin() + chosen * nr;
        std::copy(cIt, cIt + nr, dIt);
        choices[c] = chosen;
    }

    auto output = kmeans::weighted_initialization(nr, nc, data.data(), ncenters, rng);

    // Expect one entry from each category.
    EXPECT_EQ(output.size(), ncenters);
    for (auto& o : output) {
        o = choices[o];
    }
    std::sort(output.begin(), output.end());
    EXPECT_EQ(one_to_n(ncenters), output);

    // If more clusters are requested, we detect that only duplicates are available and we bail early.
    auto output2 = kmeans::weighted_initialization(nr, nc, data.data(), ncenters + 1, rng);
    EXPECT_EQ(output2.size(), ncenters);
    for (auto& o : output2) {
        o = choices[o];
    }
    std::sort(output2.begin(), output2.end());
    EXPECT_EQ(one_to_n(ncenters), output2);
}

INSTANTIATE_TEST_CASE_P(
    Initialization,
    WeightedInitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using InitializationEdgeTest = TestParamCore<std::tuple<int, int> >;

TEST_P(InitializationEdgeTest, TooManyClusters) {
    auto param = GetParam();
    assemble(param);
    std::mt19937_64 rng(nc* 10);

    auto soutput = kmeans::simple_initialization(nc, nc, rng);
    EXPECT_EQ(soutput, one_to_n(nc));

    auto soutput2 = kmeans::simple_initialization(nc, nc + 1, rng);
    EXPECT_EQ(soutput2, one_to_n(nc));

    auto woutput = kmeans::weighted_initialization(nr, nc, data.data(), nc, rng);
    std::sort(woutput.begin(), woutput.end());
    EXPECT_EQ(woutput, one_to_n(nc));

    auto woutput2 = kmeans::weighted_initialization(nr, nc, data.data(), nc + 1, rng);
    std::sort(woutput2.begin(), woutput2.end());
    EXPECT_EQ(woutput2, one_to_n(nc));
}

INSTANTIATE_TEST_CASE_P(
    Initialization,
    InitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50) // number of dimensions
    )
);
