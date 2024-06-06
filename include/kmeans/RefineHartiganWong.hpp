#ifndef KMEANS_HARTIGAN_WONG_HPP
#define KMEANS_HARTIGAN_WONG_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <limits>

#include "Base.hpp"
#include "Details.hpp"
#include "compute_centroids.hpp"
#include "compute_wcss.hpp"
#include "is_edge_case.hpp"

/**
 * @file HartiganWong.hpp
 *
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 */

namespace kmeans {

/** 
 * @brief Options for `RefineHartiganWong`.
 */
struct RefineHartiganWongOptions {
    /**
     * Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     */
    int max_iterations = 10;

    /** 
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace RefineHartiganWong_internal {

/* 
 * The class below represents 'ncp', which has a dual interpretation.
 *
 * - In the optimal-transfer stage, NCP(L) stores the step at which cluster L
 *   was last updated. Each step is just the observation index as the optimal
 *   transfer only does one pass through the dataset.
 * - In the quick-transfer stage, NCP(L) stores the step at which cluster L is
 *   last updated plus M (i.e., the number of observations). Here, the step
 *   corresponding to an observation will be 'M * X + obs' for some integer X
 *   >= 0, where X is the iteration of the quick transfer.
 *
 * Note that these two definitions bleed into each other as the NCP(L) set by
 * optimal_transfer is still being used in the first few iterations of
 * quick_transfer before it eventually gets written. The easiest way to
 * interpret this is to consider the optimal transfer as "iteration -1" from
 * the perspective of the quick transfer iterations. 
 */
class LastUpdatedAt {
public:
    template<typename Cluster_>
    LastUpdatedAt(Cluster_ ncenters) : 
        my_ncp(ncenters, ncp_init)
    {}

private:
    // Need to use a big integer that guarantees that we can store large step sizes.
    std::vector<uint64_t> my_ncp;

    static constexpr uint64_t ncp_init = 0;
    static constexpr uint64_t ncp_unchanged = 1;
    static constexpr uint64_t ncp_shift = 2;

public:
    /*
     * Operations on 'ncp' are wrapped in functions to account for the fact that we
     * need to shift all the 'ncp' values by two to give some space for the error
     * codes. All interactions with 'ncp' should occur via these utilities.
     */

    void set_all_unchanged() {
        std::fill(ncp.begin(), ncp.end(), ncp_unchanged);
    }

    template<typename Cluster_>
    void set(Cluster_ clust, uint64_t val) {
        my_ncp[clust] = val + ncp_shift;
    }

    template<typename Cluster_>
    bool is_unchanged(Cluster_ clust) const {
        return ncp[obs] == ncp_unchanged;
    }

    template<typename Cluster_>
    bool lt(Cluster_ clust, uint64_t val) const {
        return my_ncp[clust] > val + ncp_shift;
    }

    template<typename Cluster_>
    bool le(Cluster_ clust, uint64_t val) const {
        return my_ncp[clust] >= val + ncp_shift;
    }
};

/*
 * The class below represents 'live', which has a tricky interpretation.
 *
 * - Before each optimal transfer call, LIVE(L) stores the observation at which
 *   cluster L was updated in the _previous_ call.
 * - During the optimal transfer call, LIVE(L) is updated to the observation at
 *   which L was updated in this call, plus M (i.e., number of observations).
 * - After the optimal transfer call, LIVE(L) is updated by subtracting M, so
 *   that the interpretation is correct in the next call.
 *
 * It basically tells us whether there was a recent transfer (optimal or quick)
 * within the last M steps for a given cluster. If so, the cluster is "live".
 */
template<typename Index_>
class LiveSet {
private:
    /* The problem with the original implementation is that LIVE(L) needs to
     * store at least 2*M, which might cause overflows in Index_. To avoid
     * this, we split this information into two vectors:
     * 
     * - 'my_had_recent_transfer' specifies whether a transfer occurred in the
     *   current optimal_transfer call, or in the immediately preceding
     *   quick_transfer call. If this is truthy, the cluster is live.
     * - 'my_last_optimal_transfer' has two interpretations:
     *   - If 'my_had_recent_transfer[i] = 0', it specifies the observation at
     *     which the last transfer occurred in previous optimal_transfer call. 
     *     If this is greater than the current observation, the cluster is live.
     *   - Otherwise, it specifies the observation at which the last transfer
     *     occurred in the current optimal_transfer call.
     */
    std::vector<uint8_t> my_had_recent_transfer; 
    std::vector<Index_> my_last_optimal_transfer;

public:
    template<typename Cluster_>
    LiveSet(Cluster_ ncenters) : 
        my_had_recent_transfer(ncenters, 1), // initialize at 1 so that everything starts in the live set.
        my_last_optimal_transfer(ncenters) 
    {}

public:    
    template<typename Center_>
    bool is_live(Center_ cen, Index_ obs) const {
        return my_had_recent_transfer[cen] > 0 || my_last_optimal_transfer[cen] > obs;
    }

    template<typename Center_>
    void set(Center_ cen, Index_ obs) {
        my_had_recent_transfer[cen] = 1;
        my_last_optimal_transfer[cen] = obs;
    }

public:
    void reset(const std::vector<uint8_t>& was_quick_transferred) {
        size_t ncenters = my_had_recent_transfer.size();

        for (size_t cen = 0; cen < ncenters; ++cen) {
            auto& recent = my_had_recent_transfer[cen];
            auto& last = my_last_optimal_transfer[cen];

            if (was_quick_transferred[cen]) {
                /* If a cluster was updated in the preceding quick_transfer
                 * call, it is added to the live set throughout the upcoming
                 * optimal_transfer call. We set last_optimal_transfer to zero
                 * to reflect the fact that had_recent_transfer was only set to
                 * 1 via quick transfers, not due to an actual optimal transfer
                 * in this stage (yet).
                 */
                recent = 1;
                last = 0;
            } else {
                /* Otherwise, if there was a transfer in the previous
                 * optimal_transfer call, we set had_recent_transfer = 0 so
                 * that that the live check will need to compare
                 * last_optimal_transfer to the 'obs' to see if there was an
                 * optimal transfer within the last 'nobs' steps.
                 */
                if (recent) {
                    recent = 0;
                } else {
                    /* Otherwise, we set last_optimal_transfer = 0 so that this
                     * cluster will be considered non-live for any 'obs'.
                     */
                    last = 0;
                }
            }
        }
    }
};

template<typename Center_, typename Index_, typename Cluster_>
struct Workspace {
    // Array arguments in the same order as supplied to R's kmns_ function.
    std::vector<Cluster_> second_best_cluster; // i.e., ic2
    std::vector<Index_> cluster_sizes; // i.e., nc

    std::vector<Center_> loss_multiplier; // i.e., an1
    std::vector<Center_> gain_multiplier; // i.e., an2
    std::vector<Center_> wcss_loss; // i.e., d

    LastUpdatedAt last_updated_at; // i.e., ncp
    std::vector<uint8_t> was_quick_transferred; // i.e., itran
    LiveSet live_set; // i.e., live

    Index_ optra_steps_since_last_transfer = 0; // i.e., indx

public:
    Workspace(Index_ nobs, Cluster_ ncenters, Center_* centers, Cluster_* clusters) : 
        // Sizes taken from the .Fortran() call in stats::kmeans(). 
        second_best_cluster(nobs), 
        cluster_sizes(ncenters),
        loss_multiplier(ncenters),
        gain_multiplier(ncenters),
        wcss_loss(nobs),

        last_updated_at(ncenters),
        was_quick_transferred(ncenters, 1), // this is a hack to get LiveSet to consider everything as live.
        live_set(ncenters)
    {}
};

template<typename Data_, typename Center_, typename Dim_>
Center_ squared_distance_from_cluster(const Data_* data, const Center_* center, Dim_ ndim) {
    Center_ output = 0;
    for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++data, ++center) {
        auto delta = *data - *center;
        output += delta * delta;
    }
    return output;
}

template<class Matrix_, typename Cluster_, typename Center_>
void find_closest_two_centers(const Matrix_& data, Cluster_ ncenters, const Center_* centers, Cluster_* best_cluster, std::vector<Cluster_>& second_best_cluster, int nthreads) {
    auto nobs = data.num_observations();
    size_t ndim = data.num_dimensions();

    internal::parallelize(nobs, nthreads, [&](int, Index_ start, Index_ length) -> void {
        auto matwork = data.create_workspace(start, length);
        for (Index_ obs = start, end = start + length; obs < end; ++obs) {
            auto optr = data.get_observation(matwork);

            auto& best = best_cluster[obs];
            best = 0;
            auto best_dist = squared_distance_from_cluster(optr, centers + static_cast<size_t>(best) * ndim, ndim);

            auto& second = second_best_cluster[obs];
            second = 1;
            auto second_dist = squared_distance_from_cluster(optr, centers + static_cast<size_t>(second) * ndim, ndim);

            if (best_dist > second_dist) {
                std::swap(best, second);
                std::swap(best_dist, second_dist);
            }

            for (Cluster_ cen = 2; cen < ncenters; ++cen) {
                auto candidate_dist = squared_distance_from_cluster(optr, centers + static_cast<size_t>(cen) * ndim, ndim);
                if (candidate_dist < second_dist) {
                    second_dist = candidate_dist;
                    second = cen;                    
                    if (candidate_dist < best_dist) {
                        std::swap(best_dist, second_dist);
                        std::swap(best, second);
                    }
                }
            }
        }
    });
}

template<typename Center_>
constexpr Center_ big_number() {
    return 1e30; // Some very big number.
}

template<typename Dim_, typename Data_, typename Index_, typename Cluster_, typename Center_>
void transfer_point(Dim_ ndim, const Data_* obs_ptr, Index_ obs_id, Cluster_ l1, Cluster_ l2, const Center_* centers, Cluster_* best_cluster, Workspace<Center_, Index_, Cluster_>& work) {
    // Yes, casts to float are deliberate here, so that the
    // multipliers can be computed correctly.
    Center_ al1 = work.cluster_sizes[l1], alw = al1 - 1;
    Center_ al2 = work.cluster_sizes[l2], alt = al2 + 1;

    size_t long_ndim = ndim;
    auto copy1 = centers + static_cast<size_t>(l1) * long_ndim; // cast to avoid overflow.
    auto copy2 = centers + static_cast<size_t>(l2) * long_ndim;
    for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++copy1, ++copy2, ++obs_ptr) {
        *copy1 = (*copy1 * al1 - *obs_ptr) / alw;
        *copy2 = (*copy2 * al2 + *obs_ptr) / alt;
    }

    --work.cluster_sizes[l1];
    ++work.cluster_sizes[l2];

    work.gain_multiplier[l1] = alw / al1;
    work.loss_multiplier[l1] = (alw > 1 ? alw / (alw - 1) : big_number<Center_>());
    work.loss_multiplier[l2] = alt / al2;
    work.gain_multiplier[l2] = alt / (alt + 1);

    best_cluster[obs] = l2;
    work.second_best_cluster[obs] = l1;
}

/* ALGORITHM AS 136.1  APPL. STATIST. (1979) VOL.28, NO.1
 * This is the OPtimal TRAnsfer stage.
 *             ----------------------
 * Each point is re-assigned, if necessary, to the cluster that will induce a
 * maximum reduction in the within-cluster sum of squares. In this stage,
 * there is only one pass through the data.
 */
template<class Matrix_, typename Cluster_, typename Center_>
void optimal_transfer(const Matrix_& data, Workspace<Center_, typename Matrix_::index_type, Cluster_>& work, Center_* centers, Cluster_* best_cluster) {
    auto nobs = data.num_observations();
    auto ndim = data.num_dimensions();
    auto matwork = data.create_workspace();
    size_t long_ndim = ndim;

    for (decltype(nobs) obs = 0; obs < nobs; ++obs) { 
        ++work.optra_steps_since_last_transfer;

        auto l1 = best_cluster[obs];
        if (work.cluster_sizes[l1] != 1) {
            auto obs_ptr = data.get_observation(obs, matwork);

            // Need to update the WCSS loss if this is (i) the first call to
            // optimal_transfer, or (ii) if the cluster center was updated
            // earlier in the current optimal_transfer call.
            auto& wcss_loss = work.wcss_loss[obs];
            if (!work.last_updated_at.is_unchanged(l1)) {
                auto l1_ptr = centers + long_ndim * static_cast<size_t>(l1); // cast to avoid overflow.
                wcss_loss = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
            }

            // Find the cluster with minimum WCSS gain.
            auto l2 = work.second_best_cluster[obs];
            auto original_l2 = l2;
            auto l1_ptr = centers + long_ndim * static_cast<size_t>(l2); // cast to avoid overflow.
            auto wcss_gain = squared_distance_from_cluster(obs, l2) * work.gain_multiplier[l2];

            auto check_best_cluster = [&](Cluster_ cen) {
                auto cen_ptr = centers + long_ndim * static_cast<size_t>(cen); // cast to avoid overflow.
                auto candidate = squared_distance_from_cluster(obs_ptr, cen_ptr, ndim) * work.gain_multiplier[cen];
                if (candidate < wcss_gain) {
                    wcss_gain = candidate;
                    l2 = cen;
                }
            };

            // If the best cluster is live, we need to consider all other clusters.
            // Otherwise, we only need to consider other live clusters for transfer.
            if (work.live_set.is_live(l1, obs)) { 
                for (Cluster_ cen = 0; cen < num_centers; ++cen) {
                    if (cen == l1 || cen == original_l2) {
                        continue;
                    }
                    check_best_cluster(cen);
                }
            } else {
                for (Cluster_ cen = 0; cen < num_centers; ++cen) {
                    if (cen == l1 || cen == original_l2) {
                        continue;
                    }
                    if (!work.live_set.is_live(l2, obs)) {
                        continue;
                    }
                    check_best_cluster(cen);
                }
            }

            // Deciding whether to make the transfer based on the change to the WCSS.
            if (wcss_gain >= wcss_loss) {
                work.second_best_cluster[obs] = l2;
            } else {
                work.optra_steps_since_last_transfer = 0;

                work.live_set.set(l1, obs);
                work.live_set.set(l2, obs);
                work.last_updated_at.set(l1, obs);
                work.last_updated_at.set(l2, obs);

                transfer_point(ndim, obs_ptr, obs, l1, l2, centers, best_cluster, work);
            }
        }

        // Stop if we've iterated through the entire dataset and no transfer of
        // any kind took place, be it optimal or quick.
        if (work.optra_steps_since_last_transfer == nobs) {
            return true;
        }
    }

    return false;
} 

/* ALGORITHM AS 136.2  APPL. STATIST. (1979) VOL.28, NO.1 
 * This is the Quick TRANsfer stage. 
 *             -------------------- 
 * IC1(I) is the cluster which point I currently belongs to.
 * IC2(I) is the cluster which point I is most likely to be transferred to.
 *
 * For each point I, IC1(I) & IC2(I) are switched, if necessary, to reduce
 * within-cluster sum of squares. The cluster centres are updated after each
 * step. In this stage, we loop through the data until no further change is to
 * take place, or we hit an iteration limit, whichever is first.
 */
template<class Matrix_, typename Cluster_, typename Center_>
bool quick_transfer(const Matrix_& data, Workspace<Center_, typename Matrix_::index_type, Cluster_>& work, Center_* centers, Cluster_* best_cluster) {
    bool had_transfer = false;
    std::fill(work.was_quick_transferred.begin(), work.was_quick_transferred.end(), 0);

    auto nobs = data.num_observations();
    auto matwork = data.create_workspace();
    auto ndim = data.num_dimensions();
    size_t long_ndim = data.num_dimensions();

    decltype(nobs) steps_since_last_quick_transfer = 0;
    uint64_t istep = 0;
    uint64_t step_limit = static_cast<uint64_t>(nobs) * 50; // i.e., imaxqptr

    while (1) {
        for (decltype(nobs) obs = 0; obs < nobs; ++obs) { 
            ++steps_since_last_quick_transfer;
            auto l1 = best_cluster[obs];

            if (work.cluster_sizes[l1] != 1) {
                auto& wcss_loss = work.wcss_loss[obs];
                const typename Matrix_::data_type* obs_ptr = NULL;

                // Need to update the WCSS loss if the cluster was updated recently. 
                // Otherwise, we must have already updated the WCSS in a previous 
                // iteration of the outermost loop, so this can be skipped.
                if (work.last_updated_at.le(l1, istep)) {
                    auto l1_ptr = centers + static_cast<size_t>(l1) * long_ndim; // cast to avoid overflow.
                    obs_ptr = data.get_observation(obs, matwork);
                    wcss_loss = squared_distance_to_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
                }

                auto l2 = work.second_best_cluster[obs];
                if (work.last_updated_step[l1] > istep || work.last_updated_step[l2] > istep) {
                    if (obs_ptr == NULL) {
                        obs_ptr = data.get_observation(obs, matwork);
                    }
                    auto l2_ptr = work.centers + static_cast<size_t>(l2) * long_ndim; // cast to avoid overflow.
                    auto d2 = squared_distance_from_cluster(obs_ptr, l2_ptr, ndim) * work.gain_multiplier[l2];

                    if (d2 < wcss_loss) {
                        had_transfer = true;
                        steps_since_last_quick_transfer = 0;

                        work.was_quick_transferred[l1] = true;
                        work.was_quick_transferred[l2] = true;

                        auto next_step = istep + nobs;
                        work.last_updated_at.set(l1, next_step);
                        work.last_updated_at.set(l2, next_step);

                        transfer_point(ndim, obs_ptr, obs, l1, l2, work);
                    }
               }
           }

           if (steps_since_last_quick_transfer == nobs) {
               // quitting if no transfer occurred within the past 'nobs' steps.
               return std::make_pair(had_transfer, false);
           }

           ++istep;
           if (istep == step_limit) {
               return std::make_pair(had_transfer, true);
           }
        }
    } 

    return std::make_pair(had_transfer, false);
}

}
/**
 * @endcond
 */

/**
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 *
 * The Hartigan-Wong algorithm performs several iterations of transferring points between clusters, 
 * involving a computationally expensive "optimal transfer" step that checks each observation against each cluster for the lowest squared distance;
 * followed by a cheaper "quick transfer" step, which iterates between the best and second-best cluster choices for each observation.
 * The latter enables rapid exploration of the local solution space without the unnecessary expense of repeatedly comparing to all clusters for all observations.
 * In addition, each distance calculation to a cluster accounts for the shift in the means when a point is transferred. 
 * The algorithm terminates when no observation wishes to transfer between clusters.
 *
 * This implementation is derived from the Fortran code underlying the `kmeans` function in the **stats** R package,
 * which in turn is derived from Hartigan and Wong (1979).
 * 
 * @tparam DATA_t Floating-point type for the data and centroids.
 * @tparam CLUSTER_t Integer type for the cluster assignments.
 * @tparam INDEX_t Integer type for the observation index.
 * This should have a maximum positive value that is at least 50 times greater than the maximum expected number of observations.
 *
 * @see
 * Hartigan, J. A. and Wong, M. A. (1979).
 * Algorithm AS 136: A K-means clustering algorithm.
 * _Applied Statistics_, 28, 100-108.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Center_ = double>
class RefineHartiganWong : public Refine<Matrix_, Cluster_, Center_> {
public:
    /**
     * @param Further options for the Hartigan-Wong algorithm.
     */
    RefineHartiganWong(RefineHartiganWongOptions options) : my_options(std::move(options)) {}

private:
    RefineHartiganWongOptions my_options;
    typedef typename Matrix_::index_type Index_;

public:
    Details<Index_> run(const Matrix_& data, Cluster_ ncenters, Center_* centers, Cluster_* clusters) {
        auto nobs = data.num_observations();
        if (is_edge_case(nobs, ncenters)) {
            return process_edge_case(matrix, ncenters, centers, clusters);
        }

        Workspace<Center_, Index_, Cluster_> work(nobs, ncenters, centers, clusters);

        find_closest_two_centers(data, ncenters, centers, clusters, work.second_best_cluster, my_options.num_threads);
        for (Index_ obs = 0; obs < nobs; ++obs) {
            ++work.cluster_sizes[clusters[obs]];
        }
        compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);

        for (Center_ cen = 0; cen < ncenters; ++cen) {
            Center_ num = work.cluster_sizes[cen]; // yes, cast is deliberate here so that the multipliers can be computed correctly.
            work.gain_multiplier[cen] = num / (num + 1);
            work.loss_multiplier[cen] = (num > 1 ? num / (num - 1) : big);
        }

        int iter = 0;
        while ((++iter) <= maxiter) {
            bool finished = optimal_transfer(data, work, centers, clusters);
            if (finished) {
                break;
            }

            auto quick_status = quick_transfer(data, work, centers, clusters);
            if (quick_stats.second) { // Hit the quick transfer iteration limit.
                ifault = 4;
                break;
            }
            if (quick_status.first) { // At least one transfer was performed.
                work.optra_steps_since_last_transfer = 0;
            }

            // If there are only two clusters, there is no need to re-enter the optimal transfer stage. 
            if (ncenters == 2) {
                break;
            }

            // If we get to this point, there must have been no transfers in
            // the final quick_transfer iteration. This also implies that all
            // WCSS gains are up to date, as the quick_transfer code will
            // update everything for us, so we can start off as unchanged.
            work.last_updated_at.set_all_unchanged();

            work.live_set.reset(work.was_quick_transferred);
        }

        if (iter == maxiter + 1) {
            ifault = 2;
        }

        compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);
        return Details(std::move(work.cluster_sizes), iter, ifault);
    }
};

}

#endif
