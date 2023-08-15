#ifndef KMEANS_RANDOM_HPP
#define KMEANS_RANDOM_HPP

#include <vector>
#include <numeric>
#include "aarand/aarand.hpp"

namespace kmeans {

template<typename T = int, class ENGINE>
std::vector<T> sample_without_replacement(T population, size_t choose, ENGINE& eng) {
    std::vector<T> sofar;

    if (choose >= static_cast<size_t>(population)) {
        sofar.resize(population);
        std::iota(sofar.begin(), sofar.end(), 0);
    } else {
        sofar.reserve(choose);
        T traversed = 0;

        while (sofar.size() < choose) {
            if (static_cast<double>(choose - sofar.size()) > static_cast<double>(population - traversed) * aarand::standard_uniform(eng)) {
                sofar.push_back(traversed);
            }
            ++traversed;
        }
    }

    return sofar;
}

template<typename DATA_t, typename INDEX_t, class ENGINE>
INDEX_t weighted_sample(const std::vector<DATA_t>& cumulative, const std::vector<DATA_t>& mindist, INDEX_t nobs, ENGINE& eng) {
    auto total = cumulative.back();
    INDEX_t chosen_id = 0;

    do {
        const DATA_t sampled_weight = total * aarand::standard_uniform(eng);
        chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();

        // We wrap this in a do/while to defend against edge cases where
        // ties are chosen. The most obvious of these is when you get a
        // `sampled_weight` of zero _and_ there exists a bunch of zeros at
        // the start of `cumulative`. One could also get unexpected ties
        // from limited precision in floating point comparisons, so we'll
        // just be safe and implement a loop here, in the same vein as
        // uniform01.
    } while (chosen_id == nobs || mindist[chosen_id] == 0);

    return chosen_id;
}

}

#endif
