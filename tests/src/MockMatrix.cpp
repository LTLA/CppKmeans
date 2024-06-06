#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/kmeans.hpp"

using MockMatrixTest = TestCore<::testing::Test>;

TEST_F(MockMatrixTest, Basic) {
    nr = 20;
    nc = 50;
    assemble();

    kmeans::MockMatrix mock(nr, nc, data.data());
    EXPECT_EQ(mock.num_dimensions(), nr);
    EXPECT_EQ(mock.num_observations(), nc);

    auto work1 = mock.create_workspace();
    auto ptr1 = mock.get_observation(0, work1);

    auto work2 = mock.create_workspace(0, 10);
    auto ptr2 = mock.get_observation(work2);
    EXPECT_EQ(ptr1[0], ptr2[0]);

    int val = 0;
    auto work3 = mock.create_workspace(&val, 1);
    auto ptr3 = mock.get_observation(work3);
    EXPECT_EQ(ptr1[0], ptr3[0]);

    // Checking that everything compiles.
    kmeans::InitializeRandom<decltype(mock)> ir;
    kmeans::InitializeKmeanspp<decltype(mock)> kpp;
    kmeans::InitializePcaPartition<decltype(mock)> ipp;
    kmeans::RefineLloyd<decltype(mock)> ll;
    kmeans::RefineHartiganWong<decltype(mock)> hw;
    kmeans::RefineMiniBatch<decltype(mock)> mb;
}
