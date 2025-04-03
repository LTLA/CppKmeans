#ifndef KMEANS_INITIALIZE_KMEANSPP_HPP
#define KMEANS_INITIALIZE_KMEANSPP_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

#include "aarand/aarand.hpp"

#include "Initialize.hpp"
#include "Matrix.hpp"
#include "copy_into_array.hpp"
#include "parallelize.hpp"

/**
 * @file InitializeKmeanspp.hpp
 *
 * @brief Class for kmeans++ initialization.
 */

namespace kmeans {

/**
 * @brief Options for **k-means++** initialization.
 */
struct InitializeKmeansppOptions {
    /**
     * Random seed to use to construct the PRNG prior to sampling.
     */
    uint64_t seed = 6523u;

    /** 
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace InitializeKmeanspp_internal {

template<typename Float_, typename Index_, class Engine_>
Index_ weighted_sample(const std::vector<Float_>& cumulative, const std::vector<Float_>& mindist, Index_ nobs, Engine_& eng) {
    auto total = cumulative.back();
    Index_ chosen_id = 0;

    do {
        const Float_ sampled_weight = total * aarand::standard_uniform<Float_>(eng);
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

template<typename Index_, typename Float_, class Matrix_, typename Cluster_>
std::vector<Index_> run_kmeanspp(const Matrix_& data, Cluster_ ncenters, uint64_t seed, int nthreads) {
    Index_ nobs = data.num_observations();
    size_t ndim = data.num_dimensions();
    std::vector<Float_> mindist(nobs, 1);
    std::vector<Float_> cumulative(nobs);
    std::vector<Index_> sofar;
    sofar.reserve(ncenters);
    std::mt19937_64 eng(seed);

    auto last_work = data.new_extractor();
    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        if (!sofar.empty()) {
            auto last_ptr = last_work->get_observation(sofar.back());

            parallelize(nthreads, nobs, [&](int, Index_ start, Index_ length) -> void {
                auto curwork = data.new_extractor(start, length);
                for (Index_ obs = start, end = start + length; obs < end; ++obs) {
                    auto current = curwork->get_observation(); // make sure this is outside the if(), as we MUST call this in every loop iteration to fulfill consecutive access.

                    if (mindist[obs]) {
                        Float_ r2 = 0;
                        for (size_t d = 0; d < ndim; ++d) {
                            Float_ delta = static_cast<Float_>(current[d]) - static_cast<Float_>(last_ptr[d]); // cast to ensure consistent precision regardless of Data_.
                            r2 += delta * delta;
                        }

                        if (cen == 1 || r2 < mindist[obs]) {
                            mindist[obs] = r2;
                        }
                    }
                }
            });
        }

        cumulative[0] = mindist[0];
        for (Index_ i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }

        const auto total = cumulative.back();
        if (total == 0) { // a.k.a. only duplicates left.
            break;
        }

        auto chosen_id = weighted_sample(cumulative, mindist, nobs, eng);
        mindist[chosen_id] = 0;
        sofar.push_back(chosen_id);
    }

    return sofar;
}

}
/**
 * @endcond
 */

/**
 * @brief **k-means++** initialization of Arthur and Vassilvitskii (2007).
 *
 * This approach involves the selection of starting points via iterations of weighted sampling, 
 * where the sampling probability for each point is proportional to the squared distance to the closest starting point that was chosen in any of the previous iterations.
 * The aim is to obtain well-separated starting points to encourage the formation of suitable clusters.
 *
 * @tparam Index_ Integer type for the observation indices in the input dataset.
 * @tparam Data_ Numeric type for the input dataset.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * @tparam Matrix_ Class of the input data matrix.
 * This should satisfy the `Matrix` interface.
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class InitializeKmeanspp final : public Initialize<Index_, Data_, Cluster_, Float_, Matrix_> {
private:
    InitializeKmeansppOptions my_options;

public:
    /**
     * @param options Options for kmeans++ initialization.
     */
    InitializeKmeanspp(InitializeKmeansppOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializeKmeanspp() = default;

public:
    /**
     * @return Options for kmeans++ partitioning, to be modified prior to calling `run()`.
     */
    InitializeKmeansppOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Cluster_ run(const Matrix_& matrix, Cluster_ ncenters, Float_* centers) const {
        Index_ nobs = matrix.num_observations();
        if (!nobs) {
            return 0;
        }

        auto sofar = InitializeKmeanspp_internal::run_kmeanspp<Index_, Float_>(matrix, ncenters, my_options.seed, my_options.num_threads);
        internal::copy_into_array(matrix, sofar, centers);
        return sofar.size();
    }
    /**
     * @endcond
     */
};

}

#endif
