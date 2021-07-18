# Building the function.
library(Rcpp)
if (!file.exists("kmeans")) {
    file.symlink("../../include/kmeans", "kmeans")
}
sourceCpp("test.cpp")

library(testthat)
for (nc in c(20, 50, 100)) {
    for (nr in c(2, 10, 20)) {
        for (k in c(2, 5, 10)) {
            cat("NR = ", nr, ", NC = ", nc, ", k = " , k, "\n", sep="")

            # Generating some data.
            set.seed(nc / nr + k)
            mat <- matrix(rnorm(nr * nc), ncol=nc)
            centers <- sample(nc, k)
            init <- mat[,centers]
            
            out <- run_kmeans(mat, init)
            ref <- kmeans(t(mat), centers=t(init))

            expect_equal(out$centers, unname(t(ref$centers)))
            expect_identical(out$clusters + 1L, ref$cluster)
            expect_identical(out$size, ref$size)
            expect_equal(out$wcss, ref$withinss)
            expect_identical(out$status, ref$ifault)
            expect_identical(out$iterations, ref$iter)
        }
    }
}
