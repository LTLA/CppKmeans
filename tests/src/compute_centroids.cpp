#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/compute_centroids.hpp"
#include "kmeans/SimpleMatrix.hpp"

using ComputeCentroidsTest = TestParamCore<std::tuple<int, int, int> >;

TEST_P(ComputeCentroidsTest, Basic) {
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

    // Computing by row for comparison.
    std::vector<double> buffer(nc);
    std::vector<double> ref(ncenters * nr);
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            buffer[c] = data[c * nr + r];
        }

        for (int c = 0; c < nc; ++c) {
            ref[clusters[c] * nr + r] += buffer[c];
        }
        for (int c = 0; c < ncenters; ++c) {
            ref[c * nr + r] /= cluster_size[c];
        }
    }

    EXPECT_EQ(ref, centers);
}

INSTANTIATE_TEST_SUITE_P(
    ComputeCentroids,
    ComputeCentroidsTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20),
        ::testing::Values(50, 100),
        ::testing::Values(3, 7, 11) // number of clusters
    )
);

using ComputeCentroidsSingleTest = TestParamCore<std::tuple<int, int> >;

TEST_P(ComputeCentroidsSingleTest, Basic) {
    auto param = GetParam();
    assemble(param);
    kmeans::SimpleMatrix mat(nr, nc, data.data());

    std::vector<double> center(nr);
    kmeans::internal::compute_centroid(mat, center.data());

    std::vector<double> ref(nr);
    std::vector<int> clusters(nc);
    std::vector<int> cluster_size { nc };
    kmeans::internal::compute_centroids(mat, 1, ref.data(), clusters.data(), cluster_size);

    EXPECT_EQ(ref, center);
}

INSTANTIATE_TEST_SUITE_P(
    ComputeCentroids,
    ComputeCentroidsSingleTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20),
        ::testing::Values(50, 100)
    )
);
