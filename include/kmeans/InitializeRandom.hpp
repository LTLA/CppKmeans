#ifndef KMEANS_INITIALIZE_RANDOM_HPP
#define KMEANS_INITIALIZE_RANDOM_HPP 

#include <algorithm>
#include <cstdint>
#include <random>

#include "Initialize.hpp"
#include "random.hpp"

#include "aarand/aarand.hpp"

/**
 * @file InitializeRandom.hpp
 *
 * @brief Class for random initialization.
 */

namespace kmeans {


/**
 * @brief Options to use for `InitializeRandom`.
 */
struct InitializeRandomOptions {
    /**
     * Random seed to use to construct the PRNG prior to sampling.
     */
    uint64_t seed = 6523u;
};

/**
 * @brief Initialize by sampling random observations without replacement.
 *
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster index.
 * @tparam Index_ Integer type for the observation index.
 */
template<typename Data_ = double, typename Cluster_ = int, typename Index_ = int>
class InitializeRandom : public Initialize<Data_, Cluster_, Index_> { 
private:
    InitializeRandomOptions my_options;

public:
    /**
     * @param options Options for random initialization.
     */
    InitializeRandom(InitializeRandomOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializeRandom() = default;

public:
    Cluster_ run(int ndim, Index_ nobs, const Data_* data, Cluster_ ncenters, Data_* centers) {
        std::mt19937_64 eng(my_options.seed);
        std::vector<Cluster_> chosen(std::min(nobs, static_cast<Index_>(ncenters)));
        aarand::sample(nobs, ncenters, chosen.begin());
        internal::copy_into_array(chosen, ndim, data, centers);
        return chosen.size();
    }
};

}

#endif
