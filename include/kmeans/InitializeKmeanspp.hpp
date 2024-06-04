#ifndef KMEANS_INITIALIZE_KMEANSPP_HPP
#define KMEANS_INITIALIZE_KMEANSPP_HPP

#include <vector>
#include <algorithm>
#include <cstdint>

#include "Initialize.hpp"
#include "random.hpp"
#include "utils.hpp"

/**
 * @file InitializeKmeansPP.hpp
 *
 * @brief Class for **kmeans++** initialization.
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
     * This should be positive.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<class Matrix_, typename Cluster_, typename Center_>
std::vector<typename Matrix_::index_type> run_kmeanspp(const Matrix_& data, Cluster_ ncenters, uint64_t seed, int nthreads) {
    typedef typename Matrix_::data_type Data_;
    typedef typename Matrix_::index_type Index_;
    typedef typename Matrix_::dimension_type Dim_;

    auto nobs = data.num_observations();
    std::vector<Data_> mindist(nobs, 1);
    std::vector<Data_> cumulative(nobs);
    std::vector<Index_> sofar;
    sofar.reserve(ncenters);
    std::mt19937_64 eng(seed);

    auto last_work = data.create_workspace();
    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        if (!sofar.empty()) {
            auto last_ptr = data.fetch_observation(sofar.back(), last_work);

            internal::parallelize(nobs, nthreads, [&](int, Index_ start, Index_ length) {
                auto curwork = matrix.create_workspace(start, length);
                for (Index_ obs = start, end = start + length; obs < end; ++obs) {
                    if (mindist[obs]) {
                        auto acopy = data.fetch_observation(curwork);
                        auto scopy = last_ptr;
                        Data_ r2 = 0;
                        for (Dim_ dim = 0; dim < ndim; ++dim, ++acopy, ++scopy) {
                            r2 += (*acopy - *scopy) * (*acopy - *scopy);
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

        auto chosen_id = internal::weighted_sample(cumulative, mindist, nobs, eng);
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
 * @tparam Matrix_ Matrix type for the input data.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Center_ Floating-point type for the centroids.
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
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
    Cluster_ run(const Matrix_& matrix, Cluster_ ncenters, Center_* centers) {
        size_t nobs = matrix.num_observations();
        if (!nobs) {
            return 0;
        }

        auto sofar = internal::run_kmeanspp(matrix, ncenters, my_options.seed, my_options.num_threads);
        internal::copy_into_array(sofar, matrix, centers);
        return sofar.size();
    }
};

}

#endif
