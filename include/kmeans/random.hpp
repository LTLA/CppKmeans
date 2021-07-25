#ifndef KMEANS_RANDOM_HPP
#define KMEANS_RANDOM_HPP

#include <vector>
#include <numeric>

namespace kmeans {

template<class ENGINE>
double uniform01 (ENGINE& eng) {
    // Stolen from Boost.
    const double factor = 1.0 / static_cast<double>((eng.max)()-(eng.min)());
    double result;
    do {
        result = static_cast<double>(eng() - (eng.min)()) * factor;
    } while (result == 1.0);
    return result;
}

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
            if (static_cast<double>(choose - sofar.size()) > static_cast<double>(population - traversed) * uniform01(eng)) {
                sofar.push_back(traversed);
            }
            ++traversed;
        }
    }

    return sofar;

}

}

#endif
