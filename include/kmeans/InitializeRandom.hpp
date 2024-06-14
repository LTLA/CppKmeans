#ifndef KMEANS_INITIALIZE_RANDOM_HPP
#define KMEANS_INITIALIZE_RANDOM_HPP 

#include <algorithm>
#include <cstdint>
#include <random>

#include "Initialize.hpp"
#include "copy_into_array.hpp"

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
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 */
template<class Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class InitializeRandom : public Initialize<Matrix_, Cluster_, Float_> { 
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
    Cluster_ run(const Matrix_& data, Cluster_ ncenters, Float_* centers) const {
        std::mt19937_64 eng(my_options.seed);
        auto nobs = data.num_observations();
        std::vector<typename Matrix_::index_type> chosen(std::min(nobs, static_cast<typename Matrix_::index_type>(ncenters)));
        aarand::sample(nobs, ncenters, chosen.begin(), eng);
        internal::copy_into_array(data, chosen, centers);
        return chosen.size();
    }
};

}

#endif
