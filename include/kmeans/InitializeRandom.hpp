#ifndef KMEANS_INITIALIZE_RANDOM_HPP
#define KMEANS_INITIALIZE_RANDOM_HPP 

#include <algorithm>
#include <cstdint>
#include <random>

#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

#include "Initialize.hpp"
#include "copy_into_array.hpp"

/**
 * @file InitializeRandom.hpp
 *
 * @brief Class for random initialization.
 */

namespace kmeans {

/**
 * Type of the pseudo-random number generator for `InitializeRandom`.
 */
typedef std::mt19937_64 InitializeRandomRng;

/**
 * @brief Options for `InitializeRandom`.
 */
struct InitializeRandomOptions {
    /**
     * Random seed to use to construct the PRNG prior to sampling.
     */
    typename InitializeRandomRng::result_type seed = sanisizer::cap<InitializeRandomRng::result_type>(6523);
};

/**
 * @brief Initialize by sampling random observations without replacement.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class InitializeRandom final : public Initialize<Index_, Data_, Cluster_, Float_, Matrix_> { 
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
    /**
     * @return Options for random initialization.
     * This can be modified prior to calling `run()`.
     */
    InitializeRandomOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Cluster_ run(const Matrix_& data, const Cluster_ ncenters, Float_* const centers) const {
        InitializeRandomRng eng(my_options.seed);
        const auto nobs = data.num_observations();
        const I<decltype(nobs)> nchosen = sanisizer::min(nobs, ncenters);
        auto chosen = sanisizer::create<std::vector<Index_> >(nchosen);
        aarand::sample(nobs, nchosen, chosen.begin(), eng);
        internal::copy_into_array(data, chosen, centers);
        return nchosen;
    }
    /**
     * @endcond
     */
};

}

#endif
