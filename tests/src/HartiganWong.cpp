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
