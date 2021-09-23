#ifndef KMEANS_RANDOM_HPP
#define KMEANS_RANDOM_HPP

#include <vector>
#include <numeric>
#include "aarand/aarand.hpp"

namespace kmeans {

template<typename T = int, class ENGINE>
std::vector<T> sample_without_replacement(T population, size_t choose, ENGINE& eng) {
    std::vector<T> sofar;

    if (choose >= population) {
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

}

#endif
