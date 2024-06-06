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
 
    int nr, nc;
    std::vector<double> data;

protected:
    std::vector<double> create_centers(const double* raw, int k) const {
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

protected:
    struct DuplicatedMatrix {
        std::vector<double> centers;
        std::vector<int> chosen;
        std::vector<double> data;
    };

    DuplicatedMatrix create_duplicate_matrix(int ncenters) const {
        DuplicatedMatrix output;

        std::mt19937_64 rng(nc * 10);
        output.chosen.resize(ncenters);
        std::iota(output.chosen.begin(), output.chosen.end(), 0);

        auto dIt = data.begin() + ncenters * nr;
        output.centers.insert(output.centers.end(), data.begin(), dIt);
        output.data.reserve(nr * nc);
        output.data.insert(output.data.end(), data.begin(), dIt);

        for (int c = ncenters; c < nc; ++c, dIt += nr) {
            auto chosen = rng() % ncenters;
            auto cIt = output.centers.begin() + chosen * nr;
            output.data.insert(output.data.end(), cIt, cIt + nr);
            output.chosen.push_back(chosen);
        }

        return output;
    }
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
