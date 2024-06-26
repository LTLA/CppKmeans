# This script executes comparisons against the reference stats::kmeans()
# implementation. It does so by using Rcpp to build a small function that
# executes the core Hartigan-Wong functionality, and then compares the 
# results after clustering with pre-defined centers. We use a preset init
# so as to avoid problems with differences in the PRNGs between C++ and R.

# Building the function.
library(Rcpp)
if (!file.exists("kmeans")) {
    file.symlink("../../include/kmeans", "kmeans")
}
if (!file.exists("aarand")) {
    file.symlink("../../build/_deps/aarand-src/include/aarand", "aarand")
}
sourceCpp("test.cpp")

library(testthat)

test_that("Hartigan-Wong works correctly", {
    for (nc in c(20, 50, 100, 500, 2000)) {
        for (nr in c(2, 10, 20)) {
            for (k in c(2, 5, 10)) {
                cat("NR = ", nr, ", NC = ", nc, ", k = " , k, "\n", sep="")

                # Generating some data.
                set.seed(nc / nr + k)
                mat <- matrix(rnorm(nr * nc), ncol=nc)
                centers <- sample(nc, k)
                init <- mat[,centers]
                
                out <- run_kmeans_hw(mat, init)
                ref <- kmeans(t(mat), centers=t(init))

                expect_equal(out$centers, unname(t(ref$centers)))
                expect_identical(out$clusters + 1L, ref$cluster)
                expect_identical(out$size, ref$size)
                expect_identical(out$status, ref$ifault)
                expect_identical(out$iterations, ref$iter)
            }
        }
    }
})

test_that("Lloyd works correctly", {
    for (nc in c(20, 50, 100, 500, 2000)) {
        for (nr in c(2, 10, 20)) {
            for (k in c(2, 5, 10)) {
                cat("NR = ", nr, ", NC = ", nc, ", k = " , k, "\n", sep="")

                # Generating some data.
                set.seed(nc / nr + k)
                mat <- matrix(rnorm(nr * nc), ncol=nc)
                centers <- sample(nc, k)
                init <- mat[,centers]
                
                out <- run_kmeans_lloyd(mat, init)
                ref <- suppressWarnings(kmeans(t(mat), centers=t(init), algorithm="Lloyd"))

                expect_equal(out$centers, unname(t(ref$centers)))
                expect_identical(out$clusters + 1L, ref$cluster)
                expect_identical(out$size, ref$size)
                expect_identical(out$iterations, ref$iter)
            }
        }
    }
})

