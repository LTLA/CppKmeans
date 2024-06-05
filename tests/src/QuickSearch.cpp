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
}

INSTANTIATE_TEST_SUITE_P(
    QuickSearch,
    QuickSearchTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(2, 10, 50) // number of observations 
    )
);
