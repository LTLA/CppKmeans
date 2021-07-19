#ifndef KMEANS_INITIALIZATION_HPP
#define KMEANS_INITIALIZATION_HPP 

#include <vector>
#include <random>
#include <cstdint>
#include <iostream>

/**
 * @file initialization.hpp
 *
 * @brief Initialization methods to choose the starting centroids.
 */

namespace kmeans {

/**
 * @cond
 */
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
/**
 * @endcond
 */

/**
 * Implements the <b>k-means++</b> initialization described by Arthur and Vassilvitskii (2007).
 * This approach involves the selection of starting points via iterations of weighted sampling, 
 * where the sampling probability for each point is defined as the squared distance to the closest starting point that was chosen in any of the previous iterations.
 * The aim is to obtain well-separated starting points to encourage the formation of suitable clusters.
 *
 * @tparam ENGINE A random number engine, e.g., `std::mt19937`.
 *
 * @param nd Number of dimensions.
 * @param no Number of observations.
 * @param data Pointer to an array where the dimensions are rows and the observations are columns.
 * Data should be stored in column-major format.
 * @param ncenters Number of centers to pick.
 * @param eng An instance of a random number engine.
 *
 * @return A vector of indices for the observations that were selected as starting points.
 * Note that the length may be less than `ncenters` if `ncenters > no` (in which case each observation is chosen as a starting point)
 * or if there are duplicate observations (in which case only one of each set of duplicates is chosen).
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<class ENGINE>
std::vector<int> weighted_initialization(int nd, int no, const double* data, int ncenters, ENGINE& eng) {
    std::vector<int> targets(no);
    std::vector<double> mindist(no, 1);
    std::vector<double> cumulative(no);
    std::vector<uint8_t> chosen(no);
    std::vector<int> sofar;
    sofar.reserve(ncenters);

    for (int cen = 0; cen < ncenters; ++cen) {
        int counter = 0;
        if (!sofar.empty()) {
            for (int obs = 0; obs < no; ++obs) {
                if (!chosen[obs]) {
                    targets[counter] = obs;
                    double& curmin = mindist[counter];
                    curmin = -1;

                    for (auto so : sofar) {
                        const double* acopy = data + obs * nd;
                        const double* scopy = data + so * nd;
                        double r2 = 0;
                        for (int dim = 0; dim < nd; ++dim, ++acopy, ++scopy) {
                            r2 += (*acopy - *scopy) * (*acopy - *scopy);
                        }

                        if (curmin < 0 || r2 < curmin) {
                            curmin = r2;
                        }
                    }

                    ++counter;
                }
            }
        } else {
            std::iota(targets.begin(), targets.end(), 0);
            counter = no;
        }

        if (!counter) {
            break;
        }

        cumulative[0] = mindist[0];
        for (int i = 1; i < counter; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }

        const double total = cumulative[counter-1];
        if (total == 0) { // a.k.a. duplicates.
            break;
        }

        const double chosen_weight = total * uniform01(eng);
        auto chosen_pos = std::lower_bound(cumulative.begin(), cumulative.begin() + counter, chosen_weight);
        auto chosen_id = targets[chosen_pos - cumulative.begin()];
        chosen[chosen_id] = true;
        sofar.push_back(chosen_id);
    }

    return sofar;
}

/**
 * Implements a simple initialization of the starting points where random observations are sampled without replacement.
 *
 * @tparam ENGINE A random number engine, e.g., `std::mt19937`.
 *
 * @param nd Number of dimensions.
 * @param no Number of observations.
 * @param data Pointer to an array where the dimensions are rows and the observations are columns.
 * Data should be stored in column-major format.
 * @param ncenters Number of centers to pick.
 * @param eng An instance of a random number engine.
 *
 * @return A vector of indices for the observations that were selected as starting points.
 * Note that the length may be less than `ncenters` if `ncenters > no`, in which case each observation is chosen as a starting point.
 */
template<class ENGINE>
std::vector<int> simple_initialization(int nd, int no, const double* data, int ncenters, ENGINE& eng) {
    std::vector<int> sofar;

    if (ncenters >= no) {
        sofar.resize(no);
        std::iota(sofar.begin(), sofar.end(), 0);
    } else {
        sofar.reserve(ncenters);
        int traversed = 0;

        while (sofar.size() < static_cast<size_t>(ncenters)) {
            if (static_cast<double>(ncenters - sofar.size()) > static_cast<double>(no - traversed) * uniform01(eng)) {
                sofar.push_back(traversed);
            }
            ++traversed;
        }
    }

    return sofar;
}

}

#endif
