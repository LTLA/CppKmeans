#ifndef KMEANS_INITIALIZE_KMEANSPP_HPP
#define KMEANS_INITIALIZE_KMEANSPP_HPP

#include <vector>
#include <algorithm>
#include <cstdint>

#include "Base.hpp"
#include "InitializeRandom.hpp"
#include "random.hpp"

/**
 * @file InitializeKmeansPP.hpp
 *
 * @brief Class for **kmeans++** initialization.
 */

namespace kmeans {

/**
 * @brief Default parameter settings.
 */
struct InitializeKmeansppOptions {
    /**
     * Random seed to use to construct the PRNG prior to sampling.
     */
    uint64_t seed = 6523u;

    /** 
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Data_, typename Cluster_, typename Index_>
std::vector<Index_> run_kmeanspp(int ndim, Index_ nobs, const Data_* data, Cluster_ ncenters, uint64_t seed, int nthreads) {
    std::vector<Data_> mindist(nobs, 1);
    std::vector<Data_> cumulative(nobs);
    std::vector<Index_> sofar;
    sofar.reserve(ncenters);
    std::mt19937_64 eng(seed);

    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        if (!sofar.empty()) {
            auto last = sofar.back();

#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads)
#endif
            for (Index_ obs = 0; obs < nobs; ++obs) {
#else
            KMEANS_CUSTOM_PARALLEL(nobs, [&](Index_ first, Index_ end) -> void {
            for (Index_ obs = first; obs < end; ++obs) {
#endif

                if (mindist[obs]) {
                    const Data_* acopy = data + obs * ndim;
                    const Data_* scopy = data + last * ndim;
                    Data_ r2 = 0;
                    for (int dim = 0; dim < ndim; ++dim, ++acopy, ++scopy) {
                        r2 += (*acopy - *scopy) * (*acopy - *scopy);
                    }

                    if (cen == 1 || r2 < mindist[obs]) {
                        mindist[obs] = r2;
                    }
                }

#ifndef KMEANS_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif
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
 * @tparam Data_ Floating-point type for the data and centroids.
 * @tparam Cluster_ Integer type for the cluster index.
 * @tparam Index_ Integer type for the observation index.
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename Data_ = double, typename Cluster_ = int, typename Index_ = int>
class InitializeKmeanspp : public Initialize<Data_, Cluster_, Index_> {
private:
    InitializeKmeansppOptions my_options;

public:
    /**
     * @param options Options for **kmeans++** initialization.
     */
    InitializeKmeanspp(InitializeKmeansppOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializeKmeanspp() = default;

public:
    Cluster_ run(int ndim, Index_ nobs, const Data_* data, Cluster_ ncenters, Data_* centers) {
        if (!nobs) {
            return 0;
        }
        auto sofar = internal::run_kmeanspp(ndim, nobs, data, ncenters, my_options.seed, my_options.num_threads);
        internal::copy_into_array(sofar, ndim, data, centers);
        return sofar.size();
    }
};

}

#endif
