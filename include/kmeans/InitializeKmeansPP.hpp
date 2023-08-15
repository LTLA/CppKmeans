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
 * @brief Implements the **k-means++** initialization of Arthur and Vassilvitskii (2007).
 *
 * This approach involves the selection of starting points via iterations of weighted sampling, 
 * where the sampling probability for each point is proportional to the squared distance to the closest starting point that was chosen in any of the previous iterations.
 * The aim is to obtain well-separated starting points to encourage the formation of suitable clusters.
 *
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster index.
 * @tparam INDEX_t Integer type for the observation index.
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class InitializeKmeansPP : public Initialize<DATA_t, CLUSTER_t, INDEX_t> {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = 6523u;

        /** 
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param s Random seed to use to construct the PRNG prior to sampling.
     *
     * @return A reference to this `InitializeKmeansPP` object.
     */
    InitializeKmeansPP& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `InitializeKmeansPP` object.
     */
    InitializeKmeansPP& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    uint64_t seed = Defaults::seed;
    int nthreads = Defaults::num_threads;

public:
    /**
     * @cond
     */
   std::vector<INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters) {
        std::vector<DATA_t> mindist(nobs, 1);
        std::vector<DATA_t> cumulative(nobs);
        std::vector<INDEX_t> sofar;
        sofar.reserve(ncenters);
        std::mt19937_64 eng(seed);

        for (CLUSTER_t cen = 0; cen < ncenters; ++cen) {
            if (!sofar.empty()) {
                auto last = sofar.back();

#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef _OPENMP
                #pragma omp parallel for num_threads(nthreads)
#endif
                for (INDEX_t obs = 0; obs < nobs; ++obs) {
#else
                KMEANS_CUSTOM_PARALLEL(nobs, [&](INDEX_t first, INDEX_t end) -> void {
                for (INDEX_t obs = first; obs < end; ++obs) {
#endif
                    if (mindist[obs]) {
                        const DATA_t* acopy = data + obs * ndim;
                        const DATA_t* scopy = data + last * ndim;
                        DATA_t r2 = 0;
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
            for (INDEX_t i = 1; i < nobs; ++i) {
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
    /**
     * @endcond
     */

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
        if (!nobs) {
            return 0;
        }
        auto sofar = run(ndim, nobs, data, ncenters);
        copy_into_array(sofar, ndim, data, centers);
        return sofar.size();
    }
};

}

#endif
