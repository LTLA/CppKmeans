#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializePCAPartition.hpp"

TEST(PCAPartitionUtils, L2normalization) {
    std::vector<double> x{1,2,3,4};
    auto output = kmeans::InitializePCAPartition<>::normalize(4, x.data());
    EXPECT_FLOAT_EQ(output, std::sqrt(30.0));

    EXPECT_FLOAT_EQ(x[0] * 2, x[1]);
    EXPECT_FLOAT_EQ(x[0] * 3, x[2]);
    EXPECT_FLOAT_EQ(x[0] * 4, x[3]);

    double l2_again = 0;
    for (auto v : x) {
        l2_again += v * v;
    }
    EXPECT_FLOAT_EQ(l2_again, 1);
}

TEST(PCAPartitionUtils, MRSECalculations) {
    size_t nr = 7;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(1000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }

    // Subset to every 10th element.
    std::vector<int> chosen;
    for (size_t c = 7; c < nc; c += 10) {
        chosen.push_back(c);
    }

    std::vector<double> center(nr);
    auto var = kmeans::InitializePCAPartition<>::update_mrse(nr, chosen, data.data(), center.data());

    // Reference calculation with no subsets.
    std::vector<int> chosenref(chosen.size());
    std::iota(chosenref.begin(), chosenref.end(), 0);
    std::vector<double> dataref(nr * chosenref.size());
    for (size_t c = 0; c < chosen.size(); ++c) {
        auto src = data.begin() + chosen[c] * nr;
        std::copy(src, src + nr, dataref.begin() + c * nr);
    }

    std::vector<double> centerref(nr);
    auto varref = kmeans::InitializePCAPartition<>::update_mrse(nr, chosenref, dataref.data(), centerref.data());

    EXPECT_EQ(var, varref);
    EXPECT_EQ(center, centerref);

    // Checking the other center method.
    std::vector<double> anothercenter(nr);
    kmeans::InitializePCAPartition<>::compute_center(nr, chosen.size(), dataref.data(), anothercenter.data());
    EXPECT_EQ(center, anothercenter);
}

TEST(PCAPartitionUtils, PowerMethodBasic) {
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

    std::vector<int> chosen(nc);
    std::iota(chosen.begin(), chosen.end(), 0);

    std::vector<double> center(point);
    for (auto& x : center) {
        x /= 2;
    }

    kmeans::InitializePCAPartition init;
    std::mt19937_64 rng;
    auto output = init.compute_pc1(nr, chosen, data.data(), center.data(), rng);

    // Computing the expected value.
    double l2 = 0;
    for (auto x : point) {
        l2 += x * x;
    }
    l2 = std::sqrt(l2);

    for (size_t r = 0; r < nr; ++r) {
        EXPECT_FLOAT_EQ(point[r] / l2, output[r]);
    }
}

TEST(PCAPartitionUtils, PowerMethodComplex) {
    size_t nr = 5;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(1000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }

    std::vector<int> chosen(nc);
    std::iota(chosen.begin(), chosen.end(), 0);
    std::vector<double> center(nr);
    kmeans::InitializePCAPartition<>::compute_center(nr, chosen, data.data(), center.data());

    kmeans::InitializePCAPartition init;
    auto output = init.compute_pc1(nr, chosen, data.data(), center.data(), rng);

    // Checking that the variance is indeed maximized on this axis.
    auto create_projections = [&](const double* ptr) -> std::vector<double> {
        std::vector<double> projections(nc);
        for (size_t c = 0; c < nc; ++c) {
            auto dptr = data.data() + c * nr;
            double& proj = projections[c];
            for (size_t r = 0; r < nr; ++r) {
                proj += (dptr[r] - center[r]) * ptr[r];
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
    
    auto refproj = create_projections(output.data());
    auto refstat = compute_stats(refproj);
    EXPECT_TRUE(std::abs(refstat.first) < 1e-6);

    for (size_t r = 0; r < nr; ++r) { // shifting the vector slightly and checking that we get a smaller variance.
        auto modified = output;
        modified[r] *= 1.1;
        kmeans::InitializePCAPartition<>::normalize(nr, modified.data());

        auto altproj = create_projections(modified.data());
        auto altstat = compute_stats(altproj);
        EXPECT_TRUE(altstat.second < refstat.second);
    }
}

TEST(PCAPartitionUtils, PowerMethodSubsetting) {
    size_t nr = 5;
    size_t nc = 121;

    std::vector<double> data(nr * nc);
    std::mt19937_64 rng(2000);
    std::normal_distribution<> norm(0.0, 1.0);
    for (auto& d : data) {
        d = norm(rng);            
    }

    // Subset to every 10th element.
    std::vector<int> chosen;
    for (size_t c = 7; c < nc; c += 10) {
        chosen.push_back(c);
    }
    std::vector<double> center(nr);
    kmeans::InitializePCAPartition<>::compute_center(nr, chosen, data.data(), center.data());

    kmeans::InitializePCAPartition init;
    std::vector<double> output;
    {
        std::mt19937_64 rng(2000); // using a fresh rng for exact reproducibility.
        output = init.compute_pc1(nr, chosen, data.data(), center.data(), rng);
    }

    // Reference calculation with no subsets.
    std::vector<int> chosenref(chosen.size());
    std::iota(chosenref.begin(), chosenref.end(), 0);
    std::vector<double> dataref(nr * chosenref.size());
    for (size_t c = 0; c < chosen.size(); ++c) {
        auto src = data.begin() + chosen[c] * nr;
        std::copy(src, src + nr, dataref.begin() + c * nr);
    }

    {
        std::mt19937_64 rng(2000);
        auto ref = init.compute_pc1(nr, chosenref, dataref.data(), center.data(), rng);
        EXPECT_EQ(ref, output);
    }
}

using PCAPartitionInitializationTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(PCAPartitionInitializationTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    kmeans::InitializePCAPartition init;
    init.set_seed(ncenters * 10);

    std::vector<double> centers(nr * ncenters);
    std::vector<int> clusters(nc);
    auto nfilled = init.run(nr, nc, data.data(), ncenters, centers.data(), clusters.data());
    EXPECT_EQ(nfilled, ncenters);
}

TEST_P(PCAPartitionInitializationTest, Sanity) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    // Duplicating the first 'ncenters' elements over and over again.
    std::mt19937_64 rng(nc * 10);
    auto dIt = data.begin() + ncenters * nr;
    for (int c = ncenters; c < nc; ++c, dIt += nr) {
        auto chosen = rng() % ncenters;
        auto cIt = data.begin() + chosen * nr;
        std::copy(cIt, cIt + nr, dIt);
    }

    kmeans::InitializePCAPartition init;
    init.set_seed(ncenters * 100);

    std::vector<double> centers(nr * ncenters);
    std::vector<int> clusters(nc);
    auto output = init.run(nr, nc, data.data(), ncenters, centers.data(), clusters.data());
    EXPECT_EQ(output, ncenters);

    // Expect one entry from each of the first 'ncenters' elements.
    // We'll just do a brute-force search for them here.
    for (int i = 0; i < ncenters; ++i) {
        auto expected = data.begin() + i * nr;
        bool found = false;

        for (int j = 0; j < ncenters; ++j) {
            auto observed = centers.data() + j * nr;
            bool okay = true;
            for (int d = 0; d < nr; ++d) {
                if (std::abs(expected[d] - observed[d]) > 0.000001) {
                    okay = false;
                    break;
                }
            }

            if (okay) {
                found = true;
                break;
            }
        }

        EXPECT_TRUE(found);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PCAPartitionInitialization,
    PCAPartitionInitializationTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(200, 2000), // number of observations
        ::testing::Values(2, 5, 10) // number of clusters
    )
);

using PCAPartitionInitializationEdgeTest = TestParamCore<std::tuple<int, int> >;

TEST_P(PCAPartitionInitializationEdgeTest, TooManyClusters) {
    auto param = GetParam();
    assemble(param);

    kmeans::InitializePCAPartition init;
    init.set_seed(nc * 100);

    std::vector<double> centers(nr * nc);
    std::vector<int> clusters(nc);

    auto woutput = init.run(nr, nc, data.data(), nc, centers.data(), clusters.data());
    EXPECT_EQ(woutput, nc);

    centers.resize(nr * (nc + 1));
    auto woutput2 = init.run(nr, nc, data.data(), nc + 1, centers.data(), clusters.data());
    EXPECT_EQ(woutput2, nc);
}

INSTANTIATE_TEST_SUITE_P(
    PCAPartitionInitialization,
    PCAPartitionInitializationEdgeTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 50) // number of observations
    )
);
