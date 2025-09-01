# library(testthat); library(kmeans.tests); source("test-hartigan_wong.R")

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

                out <- kmeans.tests:::hartigan_wong(mat, init)
                ref <- kmeans(t(mat), centers=t(init))

                expect_equal(out$centers, unname(t(ref$centers)))
                expect_identical(out$clusters + 1L, ref$cluster)
                expect_identical(out$size, ref$size)

                if (ref$ifault != 0L) {
                    expect_identical(out$status, 2L)
                } else {
                    expect_identical(out$iterations, ref$iter)
                }
            }
        }
    }
})
