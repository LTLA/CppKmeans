#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/SimpleMatrix.hpp"

class SimpleMatrixTest : public TestCore, public ::testing::Test {
    void SetUp() {
        assemble({ 20, 50 });
    }
};

TEST_F(SimpleMatrixTest, Basic) {
    kmeans::SimpleMatrix mock(nr, nc, data.data());
    EXPECT_EQ(mock.num_dimensions(), nr);
    EXPECT_EQ(mock.num_observations(), nc);

    auto work1 = mock.new_extractor();
    auto ptr1 = work1->get_observation(0);
    EXPECT_EQ(*ptr1, data[0]);
    ptr1 = work1->get_observation(10);
    EXPECT_EQ(*ptr1, data[10 * nr]);

    auto work2 = mock.new_extractor(0, 10);
    auto ptr2 = work2->get_observation();
    EXPECT_EQ(ptr2[0], data[0]);
    ptr2 = work2->get_observation();
    EXPECT_EQ(ptr2[0], data[nr]);

    int val [2] = { 5, 15 };
    auto work3 = mock.new_extractor(val, 2);
    auto ptr3 = work3->get_observation();
    EXPECT_EQ(ptr3[0], data[nr * 5]);
    ptr3 = work3->get_observation();
    EXPECT_EQ(ptr3[0], data[nr * 15]);
}
