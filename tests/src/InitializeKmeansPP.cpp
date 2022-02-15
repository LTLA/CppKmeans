#include "kmeans/InitializeKmeansPP.hpp"
#include <random>
#include <vector>
#include "TestCore.h"

using KmeansPPInitializationTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(KmeansPPInitializationTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::InitializeKmeansPP init;
    init.set_seed(ncenters * 10);
    auto output = init.run(nr, nc, data.data(), ncenters);
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

TEST_P(KmeansPPInitializationTest, Deterministic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::InitializeKmeansPP init;
    init.set_seed(ncenters * 10);
    auto output = init.run(nr, nc, data.data(), ncenters);

    // Consistent results with the same initialization.
    auto output2 = init.run(nr, nc, data.data(), ncenters);
    EXPECT_EQ(output, output2);
    
    // Different results with a different seed (only works if num obs is reasonably larger than num centers).
    init.set_seed(ncenters * 11);
    auto output3 = init.run(nr, nc, data.data(), ncenters);
    EXPECT_NE(output, output3);
}

TEST_P(KmeansPPInitializationTest, Sanity) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    // Duplicating the first 'ncenters' elements over and over again.
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

    // Expect one entry from each of the first 'ncenters' elements;
    // all others are duplicates and should have sampling probabilities of zero.
    kmeans::InitializeKmeansPP init;
    init.set_seed(ncenters * 100);
    auto output = init.run(nr, nc, data.data(), ncenters);

    EXPECT_EQ(output.size(), ncenters);
    for (auto& o : output) {
        o = choices[o];
    }
    std::sort(output.begin(), output.end());

    std::vector<int> expected(ncenters);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(expected, output);

    // If more clusters are requested, we detect that only duplicates are available and we bail early.
    auto output2 = init.run(nr, nc, data.data(), ncenters + 1);
    EXPECT_EQ(output2.size(), ncenters);
    for (auto& o : output2) {
        o = choices[o];
    }
    std::sort(output2.begin(), output2.end());
    EXPECT_EQ(expected, output2);
}

INSTANTIATE_TEST_CASE_P(
    KmeansPPInitialization,
    KmeansPPInitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using KmeansPPInitializationEdgeTest = TestParamCore<std::tuple<int, int> >;

TEST_P(KmeansPPInitializationEdgeTest, TooManyClusters) {
    auto param = GetParam();
    assemble(param);

    std::vector<int> expected(nc);
    std::iota(expected.begin(), expected.end(), 0);

    kmeans::InitializeKmeansPP init;
    init.set_seed(nc * 100);

    auto woutput = init.run(nr, nc, data.data(), nc);
    std::sort(woutput.begin(), woutput.end());
    EXPECT_EQ(woutput, expected);

    auto woutput2 = init.run(nr, nc, data.data(), nc + 1);
    std::sort(woutput2.begin(), woutput2.end());
    EXPECT_EQ(woutput2, expected);
}

INSTANTIATE_TEST_CASE_P(
    KmeansPPInitialization,
    KmeansPPInitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50) // number of observations 
    )
);
