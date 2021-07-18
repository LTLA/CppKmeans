#ifndef TEST_CORE_H
#define TEST_CORE_H

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

template<class TEST>
class TestCore : public TEST {
protected:
    void assemble() {
        data.resize(nr * nc);
        std::mt19937_64 rng(1000);
        std::normal_distribution<> norm(0.0, 1.0);
        for (auto& d : data) {
            d = norm(rng);            
        }
        return;
    }

    std::vector<double> create_centers(const double* raw, int k) {
        std::vector<double> output(k * nr);

        std::vector<int> possible(nc), chosen(k);
        std::iota(possible.begin(), possible.end(), 0);

        std::mt19937_64 rng(k * 10);
        std::sample(possible.begin(), possible.end(), chosen.begin(), k, rng);

        double* sofar = output.data();
        for (auto c : chosen) {
            std::copy(raw + c * nr, raw + c * nr + nr, sofar);
            sofar += nr;
        }

        return output;
    }
        
    size_t nr, nc;
    std::vector<double> data;
};

template<class PARAM>
class TestParamCore : public TestCore<::testing::TestWithParam<PARAM> > {
protected:
    void assemble(const PARAM& param) {
        this->nr = std::get<0>(param);
        this->nc = std::get<1>(param);
        TestCore<::testing::TestWithParam<PARAM> >::assemble();
        return;
    }
};

#endif
