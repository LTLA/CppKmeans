#include "kmeans/HartiganWong.hpp"
#include <gtest/gtest.h>
#include <random>

template<class PARAM>
class TestCore : public ::testing::TestWithParam<PARAM> {
protected:
    void assemble(const PARAM& param) {
        nr = std::get<0>(param);
        nc = std::get<1>(param);
        data.resize(nr * nc);
        hydrate(data);
        return;
    }

    void hydrate(std::vector<double>& vec) {
        std::mt19937_64 rng(1000);
        std::normal_distribution<> norm(0.0, 1.0);
        for (auto& d : vec) {
            d = norm(rng);            
        }
        return;
    }

    size_t nr, nc;
    std::vector<double> data;
};

using HartiganWongTest = TestCore<std::tuple<int, int, int> >;

TEST_P(HartiganWongTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);

    std::vector<double> centers(ncenters * nr);
    hydrate(centers);

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

    if (ncenters < nc) {
        EXPECT_TRUE(hw.iterations() > 0);

        // Checking that the WCSS calculations are correct.
        const auto& wcss = hw.WCSS();
        for (size_t i = 0; i < ncenters; ++i) {
            if (counts[i] > 1) {
                EXPECT_TRUE(wcss[i] > 0);
            } else {
                EXPECT_EQ(wcss[i], 0);
            }
        }
    } else {
        // Checking that the averages are just equal to the points.
        EXPECT_EQ(data, centers);
    }

    EXPECT_EQ(hw.status(), 0);
}

INSTANTIATE_TEST_CASE_P(
    HartiganWong,
    HartiganWongTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(10, 100, 1000), // number of dimensions
        ::testing::Values(2, 5, 10) // number of clusters 
    )
);
