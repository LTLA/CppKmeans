#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeRandom.hpp"

TEST(RandomInitialization, Sampling) {
    int nc = 50;
    int ncenters = 20;
    std::mt19937_64 rng(1947);

    {
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
    }

    {
        std::vector<int> output(ncenters);
        aarand::sample(12, ncenters, output.data(), rng);
        auto copy = output;

        std::vector<int> expected(12);
        std::iota(expected.begin(), expected.end(), 0);
        expected.insert(expected.end(), 8, 0);

        EXPECT_EQ(expected, output);
    }
}

class RandomInitializationTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(RandomInitializationTest, Basic) {
    auto ncenters = std::get<1>(GetParam());

    // Checks that the class does the right thing.
    kmeans::InitializeRandomOptions opt;
    opt.seed = ncenters * 10;
    kmeans::InitializeRandom init(opt);

    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), ncenters, centers.data());
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
}

INSTANTIATE_TEST_SUITE_P(
    RandomInitialization,
    RandomInitializationTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(20, 200, 2000) // number of observations
        ),
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

class RandomInitializationEdgeTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 12, 40 });
    }
};

TEST_F(RandomInitializationEdgeTest, TooManyClusters) {
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
