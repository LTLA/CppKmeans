#include "Rcpp.h"
#include "kmeans/kmeans.hpp"

// [[Rcpp::export(rng=false)]]
Rcpp::List variance_partition(Rcpp::NumericMatrix x, int ncenters) {
    Rcpp::NumericMatrix output(x.nrow(), ncenters);

    kmeans::InitializeVariancePartition vp;
    kmeans::SimpleMatrix<double, int> mat(x.nrow(), x.ncol(), x.begin());
    auto count = vp.run(mat, ncenters, output.begin());

    return Rcpp::List::create(
        Rcpp::Named("centers") = output, 
        Rcpp::Named("clusters") = count
    );
}
