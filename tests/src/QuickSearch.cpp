#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/QuickSearch.hpp"

using QuickSearchTest = TestParamCore<std::tuple<int, int> >;

TEST_P(QuickSearchTest, Sweep) {
    auto param = GetParam();
    assemble(param);

    // Quick and dirty check to verify that we do get back the right identity.
    kmeans::internal::QuickSearch index(nr, nc, data.data()); 
    for (int c = 0; c < nc; ++c) {
        auto best = index.find(data.data() + c * nr);
        EXPECT_EQ(c, best);
    }

    // A more serious test with non-identical inputs.
    auto half_nc = nc/2;
    kmeans::internal::QuickSearch half_index(nr, half_nc, data.data()); 
    for (int c = half_nc; c < nc; ++c) {
        auto self = data.data() + c * nr;
        auto best = half_index.find_with_distance(self);

        double expected_dist = std::numeric_limits<double>::infinity();
        int expected_best = 0;
        for (int b = 0; b < half_nc; ++b) {
            double d2 = 0;
            auto other = data.data() + b * nr;
            for (int r = 0; r < nr; ++r) {
                double delta = other[r] - self[r];
                d2 += delta * delta;
            }
            if (d2 < expected_dist) {
                expected_best = b;
                expected_dist = d2;
            }
        }

        EXPECT_EQ(expected_best, best.first);
        EXPECT_EQ(std::sqrt(expected_dist), best.second);
    }
}

TEST_P(QuickSearchTest, TakeTwo) {
    auto param = GetParam();
    assemble(param);

    kmeans::internal::QuickSearch index(nr, nc, data.data()); 
    for (int c = 0; c < nc; ++c) {
        auto res = index.find2(data.data() + c * nr);
        EXPECT_EQ(c, res.first);

        auto self = data.data() + c * nr;
        double expected_dist = std::numeric_limits<double>::infinity();
        int expected_second = 0;
        for (int b = 0; b < nc; ++b) {
            if (b == c) {
                continue;
            }

            double d2 = 0;
            auto other = data.data() + b * nr;
            for (int r = 0; r < nr; ++r) {
                double delta = other[r] - self[r];
                d2 += delta * delta;
            }

            if (d2 < expected_dist) {
                expected_second = b;
                expected_dist = d2;
            }
        }

        EXPECT_EQ(expected_second, res.second);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuickSearch,
    QuickSearchTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(2, 10, 50) // number of observations 
    )
);
