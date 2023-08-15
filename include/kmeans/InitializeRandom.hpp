#ifndef KMEANS_INITIALIZE_RANDOM_HPP
#define KMEANS_INITIALIZE_RANDOM_HPP 

#include <algorithm>
#include <cstdint>
#include <random>

#include "Base.hpp"
#include "random.hpp"

/**
 * @file InitializeRandom.hpp
 *
 * @brief Class for random initialization.
 */

namespace kmeans {

/**
 * @cond
 */
template<class V, typename DATA_t>
void copy_into_array(const V& chosen, int ndim, const DATA_t* in, DATA_t* out) {
    for (auto c : chosen) {
        auto ptr = in + c * ndim;
        std::copy(ptr, ptr + ndim, out);
        out += ndim;
    }
    return;
}
/**
 * @endcond
 */

/**
 * @brief Initialize starting points by sampling random observations without replacement.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam INDEX_t Integer type for the observation index.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class InitializeRandom : public Initialize<DATA_t, CLUSTER_t, INDEX_t> { 
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 6523u;
    };

    /**
     * @param s Random seed to use to construct the PRNG prior to sampling.
     *
     * @return A reference to this `InitializeRandom` object.
     */
    InitializeRandom& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }
private:
    uint64_t seed = Defaults::seed;
public:
    /*
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param data Pointer to an array where the dimensions are rows and the observations are columns.
     * Data should be stored in column-major format.
     * @param ncenters Number of centers to pick.
     * @param[out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On output, this will contain the final centroid locations for each cluster.
     * Data should be stored in column-major order.
     * @param clusters Ignored in this method.
     *
     * @return `centers` is filled with the new cluster centers.
     * The number of filled centers is returned, see `Initializer::run()`.
     */
    CLUSTER_t run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t*) {
        std::mt19937_64 eng(seed);
        auto chosen = sample_without_replacement(nobs, ncenters, eng);
        copy_into_array(chosen, ndim, data, centers);
        return chosen.size();
    }
};

}

#endif
