#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeKmeanspp.hpp"

using KmeansppInitializationTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(KmeansppInitializationTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::SimpleMatrix mat(nr, nc, data.data());

    int seed = ncenters * 10;
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

    // Checks that the class does the right thing.
    kmeans::InitializeKmeansppOptions opt;
    opt.seed = seed;
    kmeans::InitializeKmeanspp init(opt);

    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(mat, ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);

    auto cptr = centers.data();
    for (auto c : output) {
        std::vector<double> obs(cptr, cptr + nr);
        cptr += nr;
        auto optr = data.data() + c * nr;
        std::vector<double> exp(optr, optr + nr);
        EXPECT_EQ(obs, exp);
    }
}

TEST_P(KmeansppInitializationTest, Sanity) {
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
    kmeans::SimpleMatrix mat(nr, nc, data.data());
    auto seed = ncenters * 100;
    auto output = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters, seed, 1);

    EXPECT_EQ(output.size(), ncenters);
    for (auto& o : output) {
        o = choices[o];
    }
    std::sort(output.begin(), output.end());

    std::vector<int> expected(ncenters);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(expected, output);

    // If more clusters are requested, we detect that only duplicates are available and we bail early.
    auto output2 = kmeans::InitializeKmeanspp_internal::run_kmeanspp(mat, ncenters + 1, seed, 1);
    EXPECT_EQ(output2.size(), ncenters);
    for (auto& o : output2) {
        o = choices[o];
    }
    std::sort(output2.begin(), output2.end());
    EXPECT_EQ(expected, output2);
}

INSTANTIATE_TEST_SUITE_P(
    KmeansppInitialization,
    KmeansppInitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using KmeansppInitializationEdgeTest = TestParamCore<std::tuple<int, int> >;

TEST_P(KmeansppInitializationEdgeTest, TooManyClusters) {
    auto param = GetParam();
    assemble(param);

    kmeans::InitializeKmeansppOptions opt;
    opt.seed = nc * 10;
    kmeans::InitializeKmeanspp init(opt);

    std::vector<double> centers(nc * nr);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc, centers.data());
    EXPECT_EQ(nfilled, nc);

    // Check that there's one representative from each cluster.
    std::vector<int> equivalence, expected;
    for (int c = 0; c < nc; ++c) {
        expected.push_back(c);

        for (int d = 0; d < nc; ++d) {
            auto cIt = centers.begin() + c * nr;
            auto dIt = data.begin() + d * nr;
            bool is_equal = true;
            for (int r = 0; r < nr; ++r, ++cIt, ++dIt) {
                if (*cIt != *dIt) {
                    is_equal = false;
                    break;
                }
            }

            if (is_equal) {
                equivalence.push_back(d);
            }
        }
    }
    std::sort(equivalence.begin(), equivalence.end());
    EXPECT_EQ(equivalence, expected);

    std::vector<double> centers2(nc * nr);
    auto nfilled2 = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc + 10, centers2.data());
    EXPECT_EQ(nfilled2, nc);
    EXPECT_EQ(centers2, centers);
}

INSTANTIATE_TEST_SUITE_P(
    KmeansppInitialization,
    KmeansppInitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50)  // number of observations 
    )
);
