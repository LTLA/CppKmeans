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
 * @brief Options to use for `InitializeRandom`.
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
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
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
     * @return Options for random initialization,
     * to be modified prior to calling `run()`.
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
        auto chosen = sanisizer::create<std::vector<Index_> >(sanisizer::min(nobs, ncenters));
        aarand::sample(nobs, ncenters, chosen.begin(), eng);
        internal::copy_into_array(data, chosen, centers);
        return chosen.size();
    }
    /**
     * @endcond
     */
};

}

#endif
