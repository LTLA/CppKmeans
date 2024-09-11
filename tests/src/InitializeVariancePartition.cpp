#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeVariancePartition.hpp"
#include "kmeans/compute_wcss.hpp"

TEST(VariancePartitionInitialization, Welford) {
    size_t nr = 7;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(1000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);
    }

    std::vector<double> means(nr), ss(nr);
    for (size_t c = 0; c < nc; ++c) {
        kmeans::InitializeVariancePartition_internal::compute_welford(nr, data.data() + nr * c, means.data(), ss.data(), static_cast<double>(c + 1));
    }

    for (size_t r = 0; r < nr; ++r) {
        double refmean = 0;
        for (size_t c = 0; c < nc; ++c) {
            refmean += data[r + c * nr];
        }
        refmean /= nc;
        EXPECT_FLOAT_EQ(refmean, means[r]);

        double ref_ss = 0;
        for (size_t c = 0; c < nc; ++c) {
            auto delta = data[r + c * nr] - refmean;
            ref_ss += delta * delta;
        }
        EXPECT_FLOAT_EQ(ref_ss, ss[r]);
    }
}

class VariancePartitionInitializationTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(VariancePartitionInitializationTest, Basic) {
    auto ncenters = std::get<1>(GetParam());
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    kmeans::InitializeVariancePartition init;
    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(mat, ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);
}

TEST_P(VariancePartitionInitializationTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_duplicate_matrix(ncenters);

    // Duplicating the first 'ncenters' elements over and over again.
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    kmeans::InitializeVariancePartition init;
    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(mat, ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);

    auto matched = match_to_data(ncenters, centers, /* tolerance = */ 1e-8);
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
    VariancePartitionInitialization,
    VariancePartitionInitializationTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(200, 2000) // number of observations
        ),
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

class VariancePartitionInitializationEdgeTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 20, 43 });
    }
};

TEST_F(VariancePartitionInitializationEdgeTest, TooManyClusters) {
    kmeans::InitializeVariancePartition init;

    std::vector<double> centers(nc * nr);
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc, centers.data());
    EXPECT_EQ(nfilled, nc);

    // Check that there's one representative from each cluster.
    auto matched = match_to_data(nc, centers);
    std::sort(matched.begin(), matched.end());
    std::vector<int> expected(nc);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(matched, expected);

    std::vector<double> centers2(nc * nr);
    auto nfilled2 = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), nc + 10, centers2.data());
    EXPECT_EQ(nfilled2, nc);
    EXPECT_EQ(centers2, centers);
}

TEST(VariancePartitionInitialization, Options) {
    kmeans::InitializeVariancePartitionOptions opt;
    opt.size_adjustment = 0;
    kmeans::InitializeVariancePartition init(opt);
    EXPECT_EQ(init.get_options().size_adjustment, 0);

    init.get_options().size_adjustment = 0.9;
    EXPECT_EQ(init.get_options().size_adjustment, 0.9);
}
