#ifndef TEST_CORE_H
#define TEST_CORE_H

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <random>
#include <cmath>

class TestCore {
protected:
    static void assemble(const std::tuple<int, int>& params) {
        if (last_params == params) {
            return;
        }
        last_params = params;

        nr = std::get<0>(params);
        nc = std::get<1>(params);
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
        std::vector<double> centers;
    };

    static KnownSimulated create_duplicate_matrix(int k) {
        KnownSimulated output;

        output.clusters.reserve(k);
        output.clusters.resize(k);
        std::iota(output.clusters.begin(), output.clusters.end(), 0);

        auto dIt = data.begin();
        output.centers.insert(output.centers.end(), dIt, dIt + k * nr);
        output.data.reserve(nr * nc);
        output.data.insert(output.data.end(), output.centers.begin(), output.centers.end());

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
            auto clust = output.clusters[c];
            for (int r = 0; r < nr; ++r, ++dIt) {
                double shift = (r == 0 ? clust * 100 : 0); // shifting the first dimension to ensure that we have separate clusters.
                output.data.push_back(*dIt + shift);
            }
        }

        output.centers.resize(k * nr);
        for (int cen = 0; cen < k; ++cen) {
            output.centers[cen * nr] = 100 * cen;
        }

        return output;
    }

protected:
    static std::vector<int> match_to_data(int ncenters, const std::vector<double>& centers, double tolerance = 0) {
        std::vector<int> found;
        found.reserve(ncenters);

        for (int c = 0; c < ncenters; ++c) {
            int num_equal = 0;
            int equal_id = 0;

            for (int d = 0; d < nc; ++d) {
                auto cIt = centers.begin() + c * nr;
                auto dIt = data.begin() + d * nr;

                bool is_equal = true;
                if (tolerance) {
                    for (int r = 0; r < nr; ++r, ++cIt, ++dIt) {
                        if (std::abs(*cIt - *dIt) > tolerance) {
                            is_equal = false;
                            break;
                        }
                    }
                } else {
                    for (int r = 0; r < nr; ++r, ++cIt, ++dIt) {
                        if (*cIt != *dIt) {
                            is_equal = false;
                            break;
                        }
                    }
                }

                if (is_equal) {
                    ++num_equal;
                    equal_id = d;
                }
            }

            found.push_back(num_equal == 1 ? equal_id : -1);
        }

        return found;
    }
};

#endif
