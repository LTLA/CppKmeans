#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/compute_wcss.hpp"
#include "kmeans/compute_centroids.hpp"
#include "kmeans/SimpleMatrix.hpp"

using ComputeWcssTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(ComputeWcssTest, Basic) {
    auto param = GetParam();
    assemble(param);
    auto ncenters = std::get<2>(param);
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    std::vector<int> clusters(nc);
    std::vector<int> cluster_size(nc);
    for (int c = 0; c < nc; ++c) {
        clusters[c] = c % ncenters;
        ++cluster_size[c];
    }

    std::vector<double> centers(ncenters * nr);
    kmeans::internal::compute_centroids(mat, ncenters, centers.data(), clusters.data(), cluster_size);
    std::vector<double> wcss(ncenters);
    kmeans::compute_wcss(mat, ncenters, centers.data(), clusters.data(), wcss.data());

    // Computing by row for comparison.
    std::vector<double> ref(ncenters);
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            auto delta = data[c * nr + r] - centers[clusters[c] * nr + r];
            ref[clusters[c]] += delta * delta;
        }
    }

    for (int c = 0; c < ncenters; ++c) {
        EXPECT_FLOAT_EQ(wcss[c], ref[c]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ComputeWcss,
    ComputeWcssTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20),
        ::testing::Values(50, 100),
        ::testing::Values(3, 7, 11) // number of clusters
    )
);
