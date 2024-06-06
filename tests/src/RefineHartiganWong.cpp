#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/RefineHartiganWong.hpp"

//using HartiganWongBasicTest = TestParamCore<std::tuple<int, int, int> >;
//
//TEST_P(HartiganWongBasicTest, Sweep) {
//    auto param = GetParam();
//    assemble(param);
//    auto ncenters = std::get<2>(param);
//
//    auto centers = create_centers(data.data(), ncenters);
//    std::vector<int> clusters(nc);
//    auto hw = kmeans::HartiganWong<>().run(nr, nc, data.data(), ncenters, centers.data(), clusters.data());
//
//    // Checking that there's the specified number of clusters, and that they're all non-empty.
//    std::vector<int> counts(ncenters);
//    for (auto c : clusters) {
//        EXPECT_TRUE(c >= 0 && c < ncenters);
//        ++counts[c];
//    }
//    EXPECT_EQ(counts, hw.sizes);
//    for (auto c : counts) {
//        EXPECT_TRUE(c > 0); 
//    }
//
//    EXPECT_TRUE(hw.iterations > 0);
//
//    // Checking that the WCSS calculations are correct.
//    const auto& wcss = hw.withinss;
//    for (int i = 0; i < ncenters; ++i) {
//        if (counts[i] > 1) {
//            EXPECT_TRUE(wcss[i] > 0);
//        } else {
//            EXPECT_EQ(wcss[i], 0);
//        }
//    }
//
//    // Checking paralleization yields the same results.
//    {
//        auto pcenters = create_centers(data.data(), ncenters);
//        std::vector<int> pclusters(nc);
//        auto phw = kmeans::HartiganWong<>().set_num_threads(3).run(nr, nc, data.data(), ncenters, pcenters.data(), pclusters.data());
//        EXPECT_EQ(pcenters, centers);
//        EXPECT_EQ(pclusters, clusters);
//        EXPECT_EQ(phw.withinss, wcss);
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    HartiganWong,
//    HartiganWongBasicTest,
//    ::testing::Combine(
//        ::testing::Values(10, 20), // number of dimensions
//        ::testing::Values(20, 200, 2000), // number of observations 
//        ::testing::Values(2, 5, 10) // number of clusters 
//    )
//);
//
//using HartiganWongConstantTest = TestParamCore<std::tuple<int, int> >;
//
//TEST_P(HartiganWongConstantTest, TooMany) {
//    auto param = GetParam();
//    assemble(param);
//
//    std::vector<double> centers(data.size());
//    std::vector<int> clusters(nc);
//    auto hw = kmeans::HartiganWong<>().run(nr, nc, data.data(), nc, centers.data(), clusters.data());
//
//    // Checking that the averages are just equal to the points.
//    EXPECT_EQ(data, centers);
//    EXPECT_EQ(hw.withinss, std::vector<double>(nc));
//    EXPECT_EQ(hw.sizes, std::vector<int>(nc, 1));
//    EXPECT_EQ(hw.iterations, 0);
//
//    std::vector<int> ref(nc);
//    std::iota(ref.begin(), ref.end(), 0);
//    EXPECT_EQ(ref, clusters);
//
//    centers.resize(nr * (nc + 1));
//    auto hw2 = kmeans::HartiganWong<>().run(nr, nc, data.data(), nc + 1, centers.data(), clusters.data());
//    EXPECT_EQ(hw2.status, 3);
//    EXPECT_EQ(clusters, ref); // unchanged.
//    EXPECT_EQ(hw2.sizes[nc], 0); // last element is now zero.
//}
//
//TEST_P(HartiganWongConstantTest, TooFew) {
//    auto param = GetParam();
//    assemble(param);
//
//    std::vector<double> centers(nr);
//    std::vector<int> clusters(nc);
//    auto hw = kmeans::HartiganWong<>().run(nr, nc, data.data(), 1, centers.data(), clusters.data());
//
//    std::vector<double> averages(nr);
//    size_t i = 0;
//    for (auto d : data) {
//        averages[i] += d;
//        ++i;
//        i %= nr;
//    }
//    for (auto& a : averages) {
//        a /= nc;
//    }
//
//    EXPECT_EQ(centers, averages);
//    EXPECT_EQ(hw.iterations, 0);
//
//    // no points at all.
//    auto hw2 = kmeans::HartiganWong<>().run(nr, nc, data.data(), 0, centers.data(), clusters.data());
//    EXPECT_EQ(hw2.status, 3);
//    EXPECT_EQ(hw2.sizes, std::vector<int>());
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    HartiganWong,
//    HartiganWongConstantTest,
//    ::testing::Combine(
//        ::testing::Values(10, 20), // number of dimensions
//        ::testing::Values(20, 200, 2000) // number of observations 
//    )
//);
