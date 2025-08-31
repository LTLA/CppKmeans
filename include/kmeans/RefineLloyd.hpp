#ifndef KMEANS_LLOYD_HPP
#define KMEANS_LLOYD_HPP

#include <vector>
#include <algorithm>

#include "sanisizer/sanisizer.hpp"

#include "Refine.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "is_edge_case.hpp"
#include "compute_centroids.hpp"
#include "parallelize.hpp"

/**
 * @file RefineLloyd.hpp
 *
 * @brief Implements the Lloyd algorithm for k-means clustering.
 */

namespace kmeans {

/**
 * @brief Options for `RefineLloyd` construction.
 */
struct RefineLloydOptions {
    /**
     * Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     */
    int max_iterations = 10;

    /**
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Implements the Lloyd algorithm for k-means clustering.
 *
 * The Lloyd algorithm is the simplest k-means clustering algorithm,
 * involving several iterations of batch assignments and center calculations.
 * Specifically, we assign each observation to its closest cluster, and once all points are assigned, we recompute the cluster centroids.
 * This is repeated until there are no reassignments or the maximum number of iterations is reached.
 *
 * In the `Details::status` returned by `run()`, the status code is either 0 (success) or 2 (maximum iterations reached without convergence).
 * Previous versions of the library would report a status code of 1 upon encountering an empty cluster, but these are now just ignored.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the data.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 * This will also be used for any internal distance calculations.
 * @tparam Matrix_ Type for the input data matrix.
 * This should satisfy the `Matrix` interface.
 *
 * @see
 * Lloyd, S. P. (1982).  
 * Least squares quantization in PCM.
 * _IEEE Transactions on Information Theory_ 28, 128-137.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, typename Matrix_ = Matrix<Index_, Data_> >
class RefineLloyd final : public Refine<Index_, Data_, Cluster_, Float_, Matrix_> {
private:
    RefineLloydOptions my_options;

public:
    /**
     * @param options Further options to the Lloyd algorithm.
     */
    RefineLloyd(RefineLloydOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor. 
     */
    RefineLloyd() = default;

public:
    /**
     * @return Options for Lloyd clustering,
     * to be modified prior to calling `run()`.
     */
    RefineLloydOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Details<Index_> run(const Matrix_& data, const Cluster_ ncenters, Float_* const centers, Cluster_* const clusters) const {
        const auto nobs = data.num_observations();
        if (internal::is_edge_case(nobs, ncenters)) {
            return internal::process_edge_case(data, ncenters, centers, clusters);
        }

        auto sizes = sanisizer::create<std::vector<Index_> >(ncenters);
        auto copy = sanisizer::create<std::vector<Cluster_> >(nobs);

        const auto ndim = data.num_dimensions();
        internal::QuickSearch<Float_, Cluster_> index;

        decltype(I(my_options.max_iterations)) iter = 1;
        for (; iter <= my_options.max_iterations; ++iter) {
            index.reset(ndim, ncenters, centers);
            parallelize(my_options.num_threads, nobs, [&](const int, const Index_ start, const Index_ length) -> void {
                auto work = data.new_extractor(start, length);
                for (Index_ obs = start, end = start + length; obs < end; ++obs) {
                    const auto dptr = work->get_observation();
                    copy[obs] = index.find(dptr); 
                }
            });

            // Checking if it already converged.
            bool updated = false;
            for (Index_ obs = 0; obs < nobs; ++obs) {
                if (copy[obs] != clusters[obs]) {
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                break;
            }
            std::copy(copy.begin(), copy.end(), clusters);

            std::fill(sizes.begin(), sizes.end(), 0);
            for (Index_ obs = 0; obs < nobs; ++obs) {
                ++sizes[clusters[obs]];
            }
            internal::compute_centroids(data, ncenters, centers, clusters, sizes);
        }

        int status = 0;
        if (iter == my_options.max_iterations + 1) {
            status = 2;
        }
        return Details<Index_>(std::move(sizes), iter, status);
    }
    /**
     * @endcond
     */
};

}

#endif
