#include "kmeans/HartiganWong.hpp"
#include <gtest/gtest.h>
#include "TestCore.h"

using HartiganWongBasicTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(HartiganWongBasicTest, Sweep) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    auto centers = create_centers(data.data(), ncenters);
    kmeans::HartiganWong hw(nr, nc, data.data(), ncenters, centers.data());

    // Checking that there's the specified number of clusters, and that they're all non-empty.
    std::vector<int> counts(ncenters);
    const auto& clusters = hw.clusters();
    for (auto c : clusters) {
        EXPECT_TRUE(c >= 0 && c < ncenters);
        ++counts[c];
    }
    EXPECT_EQ(counts, hw.sizes());
    for (auto c : counts) {
        EXPECT_TRUE(c > 0); 
    }

    EXPECT_TRUE(hw.iterations() > 0);

    // Checking that the WCSS calculations are correct.
    const auto& wcss = hw.withinss();
    for (size_t i = 0; i < ncenters; ++i) {
        if (counts[i] > 1) {
            EXPECT_TRUE(wcss[i] > 0);
        } else {
            EXPECT_EQ(wcss[i], 0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    HartiganWong,
    HartiganWongBasicTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000), // number of observations 
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);

using HartiganWongConstantTest = TestParamCore<std::tuple<int, int> >;

TEST_P(HartiganWongConstantTest, TooMany) {
    auto param = GetParam();
    assemble(param);

    auto centers = data;
    kmeans::HartiganWong hw(nr, nc, data.data(), nc, centers.data());

    // Checking that the averages are just equal to the points.
    EXPECT_EQ(data, centers);
    EXPECT_EQ(hw.withinss(), std::vector<double>(nc));
    EXPECT_EQ(hw.iterations(), 0);
}

TEST_P(HartiganWongConstantTest, TooFew) {
    auto param = GetParam();
    assemble(param);

    std::vector<double> centers(nr);
    kmeans::HartiganWong hw(nr, nc, data.data(), 1, centers.data());

    std::vector<double> averages(nr);
    size_t i = 0;
    for (auto d : data) {
        averages[i] += d;
        ++i;
        i %= nr;
    }
    for (auto& a : averages) {
        a /= nc;
    }

    EXPECT_EQ(centers, averages);
    EXPECT_EQ(hw.iterations(), 0);
}

INSTANTIATE_TEST_CASE_P(
    HartiganWong,
    HartiganWongConstantTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(20, 200, 2000) // number of observations 
    )
);

using HartiganWongDeathTest = TestCore<::testing::Test>;

TEST_F(HartiganWongDeathTest, AssertionFails) {
    nr = 5;
    nc = 10;
    assemble();

    try {
        kmeans::HartiganWong(nr, nc, data.data(), 0, NULL);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).find("positive") != std::string::npos);
    }

    auto centers = create_centers(data.data(), nc + 1);
    try {
        kmeans::HartiganWong(nr, nc, data.data(), nc + 1, centers.data());
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).find("less than") != std::string::npos);
    }
}
