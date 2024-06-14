#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializePcaPartition.hpp"
#include "kmeans/compute_wcss.hpp"

TEST(PcaPartitionUtils, MRSECalculations) {
    size_t nr = 7;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(1000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }

    // Subset to every 10th element.
    std::vector<size_t> chosen;
    for (size_t c = 7; c < nc; c += 10) {
        chosen.push_back(c);
    }

    std::vector<double> center(nr);
    auto var = kmeans::InitializePcaPartition_internal::update_center_and_mrse(kmeans::SimpleMatrix(nr, nc, data.data()), chosen, center.data());
    EXPECT_TRUE(var > 0);

    // Reference calculation with no subsets.
    std::vector<size_t> chosenref(chosen.size());
    std::iota(chosenref.begin(), chosenref.end(), 0);
    std::vector<double> dataref(nr * chosenref.size());
    for (size_t c = 0; c < chosen.size(); ++c) {
        auto src = data.begin() + chosen[c] * nr;
        std::copy(src, src + nr, dataref.begin() + c * nr);
    }

    kmeans::SimpleMatrix submat(nr, chosen.size(), dataref.data());
    std::vector<double> centerref(nr);
    auto varref = kmeans::InitializePcaPartition_internal::update_center_and_mrse(submat, chosenref, centerref.data());

    EXPECT_EQ(var, varref);
    EXPECT_EQ(center, centerref);

    // Checking the other center method.
    { 
        std::vector<double> anothercenter(nr);
        kmeans::internal::compute_centroid(submat, anothercenter.data());
        EXPECT_EQ(center, anothercenter);
    }

    // Cross-checking against the WCSS calculations.
    {
        std::vector<double> wcss(1);
        std::vector<int> clusters(nc);
        kmeans::compute_wcss(submat, 1, centerref.data(), clusters.data(), wcss.data());
        EXPECT_FLOAT_EQ(var, wcss[0]/chosen.size());
    }
}

TEST(PcaPartitionUtils, PowerMethodBasic) {
    size_t nr = 3;
    size_t nc = 10;

    std::vector<double> point(nr);
    std::iota(point.rbegin(), point.rend(), 1);

    // Simulating a basic scenario.
    std::vector<double> data(nr * nc);
    for (size_t i = 0; i < nc/2; ++i) {
        auto ptr = data.data() + nr * i;
        std::fill(ptr, ptr + nr, 0);
        std::copy(point.begin(), point.end(), ptr + nr);
    }

    std::vector<size_t> chosen(nc);
    std::iota(chosen.begin(), chosen.end(), 0);

    std::vector<double> center(point);
    for (auto& x : center) {
        x /= 2;
    }

    std::mt19937_64 rng;
    kmeans::InitializePcaPartition_internal::Workspace<double> work(nr);
    kmeans::InitializePcaPartition_internal::compute_pc1(kmeans::SimpleMatrix(nr, nc, data.data()), chosen, center.data(), rng, work, powerit::Options());

    // Computing the expected value.
    double l2 = 0;
    for (auto x : point) {
        l2 += x * x;
    }
    l2 = std::sqrt(l2);

    for (size_t r = 0; r < nr; ++r) {
        EXPECT_FLOAT_EQ(point[r] / l2, work.pc[r]);
    }
}

TEST(PcaPartitionUtils, PowerMethodComplex) {
    size_t nr = 5;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(1000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    std::vector<size_t> chosen(nc);
    std::iota(chosen.begin(), chosen.end(), 0);
    std::vector<double> center(nr);
    kmeans::InitializePcaPartition_internal::compute_center(mat, chosen, center.data());

    kmeans::InitializePcaPartition_internal::Workspace<double> work(nr);
    kmeans::InitializePcaPartition_internal::compute_pc1(mat, chosen, center.data(), rng, work, powerit::Options());

    // Checking that the variance is indeed maximized on this axis.
    auto create_projections = [&](const std::vector<double>& axis) -> std::vector<double> {
        std::vector<double> projections(nc);
        for (size_t c = 0; c < nc; ++c) {
            auto dptr = data.data() + c * nr;
            double& proj = projections[c];
            for (size_t r = 0; r < nr; ++r) {
                proj += (dptr[r] - center[r]) * axis[r];
            }
        }
        return projections;
    };

    auto compute_stats = [&](const std::vector<double>& proj) -> std::pair<double, double> {
        double mean = std::accumulate(proj.begin(), proj.end(), 0.0);
        mean /= proj.size();
        double var = 0;
        for (auto& p : proj) {
            var += (p - mean) * (p - mean);
        }
        return std::make_pair(mean, var);
    };
    
    auto refproj = create_projections(work.pc);
    auto refstat = compute_stats(refproj);
    EXPECT_TRUE(std::abs(refstat.first) < 1e-6);

    for (size_t r = 0; r < nr; ++r) { // shifting the vector slightly and checking that we get a smaller variance.
        auto modified = work.pc;
        modified[r] *= 1.1;

        double l2 = 0;
        for (auto x : modified) {
            l2 += x * x;
        }
        l2 = std::sqrt(l2);
        for (auto& x : modified) {
            x /= l2;
        }

        auto altproj = create_projections(modified);
        auto altstat = compute_stats(altproj);
        EXPECT_TRUE(altstat.second < refstat.second);
    }
}

TEST(PcaPartitionUtils, PowerMethodSubsetting) {
    size_t nr = 5;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(2000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    // Subset to every 10th element.
    std::vector<size_t> chosen;
    for (size_t c = 7; c < nc; c += 10) {
        chosen.push_back(c);
    }
    std::vector<double> center(nr);
    kmeans::InitializePcaPartition_internal::compute_center(mat, chosen, center.data());
    kmeans::InitializePcaPartition_internal::Workspace<double> work(nr);
    {
        std::mt19937_64 rng(2001); // using a fresh rng for exact reproducibility.
        kmeans::InitializePcaPartition_internal::compute_pc1(mat, chosen, center.data(), rng, work, powerit::Options());
    }

    // Reference calculation with no subsets.
    std::vector<size_t> chosenref(chosen.size());
    std::iota(chosenref.begin(), chosenref.end(), 0);
    std::vector<double> dataref(nr * chosenref.size());
    for (size_t c = 0; c < chosen.size(); ++c) {
        auto src = data.begin() + chosen[c] * nr;
        std::copy(src, src + nr, dataref.begin() + c * nr);
    }

    kmeans::InitializePcaPartition_internal::Workspace<double> work2(nr);
    {
        std::mt19937_64 rng(2001);
        kmeans::InitializePcaPartition_internal::compute_pc1(kmeans::SimpleMatrix(nr, chosenref.size(), dataref.data()), chosenref, center.data(), rng, work2, powerit::Options());
    }
    EXPECT_EQ(work.pc, work2.pc);
}

class PcaPartitionInitializationTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(PcaPartitionInitializationTest, Basic) {
    auto ncenters = std::get<1>(GetParam());
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    kmeans::InitializePcaPartitionOptions opt;
    opt.seed = ncenters * 10;
    kmeans::InitializePcaPartition init(opt);

    std::vector<double> centers(nr * ncenters);
    auto nfilled = init.run(mat, ncenters, centers.data());
    EXPECT_EQ(nfilled, ncenters);
}

TEST_P(PcaPartitionInitializationTest, Sanity) {
    auto ncenters = std::get<1>(GetParam());
    auto dups = create_duplicate_matrix(ncenters);

    // Duplicating the first 'ncenters' elements over and over again.
    kmeans::SimpleMatrix mat(nr, nc, dups.data.data());

    kmeans::InitializePcaPartitionOptions opt;
    opt.seed = ncenters * 10;
    kmeans::InitializePcaPartition init(opt);

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
    PcaPartitionInitialization,
    PcaPartitionInitializationTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 20), // number of dimensions
            ::testing::Values(200, 2000) // number of observations
        ),
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

class PcaPartitionInitializationEdgeTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 20, 43 });
    }
};

TEST_F(PcaPartitionInitializationEdgeTest, TooManyClusters) {
    kmeans::InitializePcaPartition init;

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

TEST(PcaPartitionInitialization, Options) {
    kmeans::InitializePcaPartitionOptions opt;
    opt.seed = 12345;
    kmeans::InitializePcaPartition init(opt);
    EXPECT_EQ(init.get_options().seed, 12345);

    init.get_options().seed = 99999;
    EXPECT_EQ(init.get_options().seed, 99999);
}
