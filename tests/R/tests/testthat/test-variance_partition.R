# library(testthat); library(kmeans.tests); source("test-variance_partition.R")

test_that("VarPart works correctly", {
    for (nc in c(20, 50, 100, 500, 2000)) {
        for (nr in c(2, 10, 20)) {
            for (k in c(2, 5, 10)) {
                cat("NR = ", nr, ", NC = ", nc, ", k = " , k, "\n", sep="")

                # Generating some data.
                set.seed(nc / nr + k)
                mat <- matrix(rnorm(nr * nc), ncol=nc)

                out <- kmeans.tests:::variance_partition(mat, k)
                expect_equal(out$clusters, k)

                ref <- refVariancePartition(mat, k)
                expect_equal(out$centers, ref)
            }
        }
    }
})
