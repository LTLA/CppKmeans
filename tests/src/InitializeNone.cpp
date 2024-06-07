#include "TestCore.h"

#include <random>
#include <vector>

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/InitializeNone.hpp"

class InitializeNoneTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 50, 20 });
    }
};

TEST_F(InitializeNoneTest, Basic) {
    std::vector<double> centers = create_centers(3);
    auto original = centers;

    kmeans::InitializeNone init;
    auto nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), 3, centers.data());
    EXPECT_EQ(nfilled, 3);
    EXPECT_EQ(original, centers);

    nfilled = init.run(kmeans::SimpleMatrix(nr, nc, data.data()), 100, centers.data());
    EXPECT_EQ(nfilled, nc);
}
