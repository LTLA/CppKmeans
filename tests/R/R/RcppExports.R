# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

hartigan_wong <- function(x, init) {
    .Call('_kmeans_tests_hartigan_wong', PACKAGE = 'kmeans.tests', x, init)
}

lloyd <- function(x, init) {
    .Call('_kmeans_tests_lloyd', PACKAGE = 'kmeans.tests', x, init)
}

variance_partition <- function(x, ncenters) {
    .Call('_kmeans_tests_variance_partition', PACKAGE = 'kmeans.tests', x, ncenters)
}

