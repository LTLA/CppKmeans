#include "Rcpp.h"
#include "kmeans/HartiganWong.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::List run_kmeans(Rcpp::NumericMatrix x, Rcpp::NumericMatrix init) {
    Rcpp::NumericMatrix output = Rcpp::clone(init);
    Rcpp::IntegerVector clusters(x.ncol());
    auto hw = kmeans::HartiganWong().run(x.nrow(), x.ncol(), x.begin(), output.ncol(), output.begin(), clusters.begin());
    return Rcpp::List::create(
        Rcpp::Named("centers") = output, 
        Rcpp::Named("clusters") = clusters,
        Rcpp::Named("size") = Rcpp::wrap(hw.sizes),
        Rcpp::Named("wcss") = Rcpp::wrap(hw.withinss),
        Rcpp::Named("status") = hw.status,
        Rcpp::Named("iterations") = hw.iterations
    );
}
