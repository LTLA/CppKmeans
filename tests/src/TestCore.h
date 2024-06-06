#ifndef TEST_CORE_H
#define TEST_CORE_H

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <random>

class TestCore {
protected:
    static void assemble(const std::tuple<int, int>& params) {
        if (last_params == params) {
            return;
        }
        last_params = params;

        nr = std::get<0>(params);
        nc = std::get<0>(params);
        data.resize(nr * nc);

        std::mt19937_64 rng(nr * 100 + nc);
        std::normal_distribution<> norm(0.0, 1.0);
        for (auto& d : data) {
            d = norm(rng);            
        }
    }

    inline static std::tuple<int, int> last_params;
    inline static int nr, nc;
    inline static std::vector<double> data;

protected:
    static std::vector<double> create_centers(int k) {
        std::vector<double> output(k * nr);
        std::mt19937_64 rng(k * 10 + nr);
        std::normal_distribution<> norm(0.0, 1.0);
        for (auto& o : output) {
            o = norm(rng);
        }
        return output;
    }

protected:
    struct KnownSimulated {
        std::vector<int> clusters;
        std::vector<double> data;
    };

    static KnownSimulated create_duplicate_matrix(int k) {
        KnownSimulated output;
        output.clusters.reserve(k);

        output.clusters.resize(k);
        std::iota(output.clusters.begin(), output.clusters.end(), 0);

        output.data.reserve(nr * nc);
        output.data.insert(output.data.end(), data.begin(), data.begin() + k * nr);

        std::mt19937_64 rng(k * 5 + nr);
        for (int c = k; c < nc; ++c) {
            auto chosen = rng() % k;
            auto cIt = output.data.begin() + chosen * nr;
            output.data.insert(output.data.end(), cIt, cIt + nr);
            output.clusters.push_back(chosen);
        }

        return output;
    }

    static KnownSimulated create_jittered_matrix(int k) {
        KnownSimulated output;
        output.clusters.reserve(nc);
        output.data.reserve(nr * nc);

        output.clusters.resize(k);
        std::iota(output.clusters.begin(), output.clusters.end(), 0);
        std::mt19937_64 rng(k * 10 + nr);
        for (int c = k; c < nc; ++c) {
            auto chosen = rng() % k;
            output.clusters.push_back(chosen);
        }

        auto dIt = data.begin();
        for (int c = 0; c < nc; ++c) {
            auto shift = output.clusters[c] * 100;
            for (int r = 0; r < nr; ++r) {
                output.data.push_back(*dIt + shift);
            }
        }

        return output;
    }
};

#endif
