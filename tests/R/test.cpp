#include "Rcpp.h"
#include "kmeans/RefineHartiganWong.hpp"
#include "kmeans/RefineLloyd.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::List run_kmeans_hw(Rcpp::NumericMatrix x, Rcpp::NumericMatrix init) {
    Rcpp::NumericMatrix output = Rcpp::clone(init);
    Rcpp::IntegerVector clusters(x.ncol());

    kmeans::RefineHartiganWong hw;
    kmeans::SimpleMatrix<double, int> mat(x.nrow(), x.ncol(), x.begin());
    auto res = hw.run(mat, output.ncol(), output.begin(), clusters.begin());

    return Rcpp::List::create(
        Rcpp::Named("centers") = output, 
        Rcpp::Named("clusters") = clusters,
        Rcpp::Named("size") = Rcpp::wrap(res.sizes),
        Rcpp::Named("status") = res.status,
        Rcpp::Named("iterations") = res.iterations
    );
}

// [[Rcpp::export(rng=false)]]
Rcpp::List run_kmeans_lloyd(Rcpp::NumericMatrix x, Rcpp::NumericMatrix init) {
    Rcpp::NumericMatrix output = Rcpp::clone(init);
    Rcpp::IntegerVector clusters(x.ncol());

    kmeans::RefineLloyd ll;
    kmeans::SimpleMatrix<double, int> mat(x.nrow(), x.ncol(), x.begin());
    auto res = ll.run(mat, output.ncol(), output.begin(), clusters.begin());

    return Rcpp::List::create(
        Rcpp::Named("centers") = output, 
        Rcpp::Named("clusters") = clusters,
        Rcpp::Named("size") = Rcpp::wrap(res.sizes),
        Rcpp::Named("status") = res.status,
        Rcpp::Named("iterations") = res.iterations
    );
}
