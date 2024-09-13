#ifndef KMEANS_HARTIGAN_WONG_HPP
#define KMEANS_HARTIGAN_WONG_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

#include "Refine.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "parallelize.hpp"
#include "compute_centroids.hpp"
#include "is_edge_case.hpp"

/**
 * @file RefineHartiganWong.hpp
 *
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 */

namespace kmeans {

/** 
 * @brief Options for `RefineHartiganWong`.
 */
struct RefineHartiganWongOptions {
    /**
     * Maximum number of optimal transfer iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     */
    int max_iterations = 10;

    /**
     * Maximum number of quick transfer iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     */
    int max_quick_transfer_iterations = 50;

    /**
     * Whether to quit early when the number of quick transfer iterations exceeds `RefineHartiganWongOptions::max_quick_tranfer_iterations`.
     * Setting this to true mimics the default behavior of R's `kmeans()` implementation.
     */
    bool quit_on_quick_transfer_convergence_failure = false;

    /** 
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace RefineHartiganWong_internal {

/* 
 * The class below represents 'ncp', which has a dual interpretation in the
 * original Fortran implementation:
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
 *
 * In short, this data structure specifies whether a cluster was modified
 * within the last M steps. This counts steps in both optimal_transfer and
 * quick_transfer, and considers modifications from both calls.
 */
template<typename Index_>
class UpdateHistory {
private:
    /* 
     * The problem with the original implementation is that the integers are
     * expected to hold 'max_quick_iterations * M'. For a templated integer
     * type, that might not be possible, so instead we split it into two
     * vectors; one holds the last iteration at which the cluster was modified,
     * the other holds the last observation used in the modification.
     */
    Index_ my_last_observation = 0;

    int my_last_iteration = init; 

    static constexpr int init = -3;
    static constexpr int unchanged = -2;

public:
    void set_unchanged() {
        my_last_observation = unchanged;
    }

    // We treat the optimal_transfer as "iteration -1" here.
    void set_optimal(Index_ obs) {
        my_last_iteration = -1;
        my_last_observation = obs;
    }

    // Here, iter should be from '[0, max_quick_transfer_iterations)'.
    void set_quick(int iter, Index_ obs) {
        my_last_iteration = iter;
        my_last_observation = obs;
    }

public:
    bool is_unchanged() const {
        return my_last_iteration == unchanged;
    }

public:
    bool changed_after(int iter, Index_ obs) const {
        if (my_last_iteration == iter) {
            return my_last_observation > obs;
        } else {
            return my_last_iteration > iter;
        }
    }

    bool changed_after_or_at(int iter, Index_ obs) const {
        if (my_last_iteration == iter) {
            return my_last_observation >= obs;
        } else {
            return my_last_iteration > iter;
        }
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
 * within the last M steps of optimal_transfer. If so, the cluster is "live".
 */
template<typename Index_>
class LiveStatus {
private:
    enum class Event : uint8_t { NONE, PAST_OPT, CURRENT_OPT, QUICK, INIT };

    /* The problem with the original implementation is that LIVE(L) needs to
     * store at least 2*M, which might cause overflows in Index_. To avoid
     * this, we split this information into two vectors:
     * 
     * - 'my_had_recent_transfer' specifies specifies whether a transfer
     *   occurred in the current optimal_transfer call, or in the immediately
     *   preceding quick_transfer call. If this > PAST_OPT, the cluster is 
     *   definitely live; if it is == PAST_OPT, it may or may not be live.
     * - 'my_last_optimal_transfer' has two interpretations:
     *   - If 'my_had_recent_transfer == PAST_OPT', it specifies the
     *     observation at which the last transfer occurred in previous
     *     optimal_transfer call. If this is greater than the current
     *     observation, the cluster is live.
     *   - If 'my_had_recent_transfer == CURRENT_OPT', it specifies the
     *     observation at which the last transfer occurred in the current
     *     optimal_transfer call.
     *   - Otherwise it is undefined and should not be used.
     *
     * One might think that 'LiveStatus::my_last_optimal_transfer' is redundant
     * with 'UpdateHistory::my_last_observation', but the former only tracks
     * optimal transfers while the latter includes quick transfers. 
     */
    Event my_had_recent_transfer = Event::INIT; 
    Index_ my_last_optimal_transfer = 0;

public:    
    bool is_live(Index_ obs) const {
        if (my_had_recent_transfer == Event::PAST_OPT) {
            return my_last_optimal_transfer > obs;
        } else {
            return my_had_recent_transfer > Event::PAST_OPT;
        }
    }

    void mark_current(Index_ obs) {
        my_had_recent_transfer = Event::CURRENT_OPT;
        my_last_optimal_transfer = obs;
    }

    void reset(bool was_quick_transferred) {
        if (was_quick_transferred) {
            my_had_recent_transfer = Event::QUICK;
        } else if (my_had_recent_transfer == Event::CURRENT_OPT) {
            my_had_recent_transfer = Event::PAST_OPT;
        } else {
            my_had_recent_transfer = Event::NONE;
        }
    }
};

template<typename Float_, typename Index_, typename Cluster_>
struct Workspace {
    // Array arguments in the same order as supplied to R's kmns_ function.
    std::vector<Cluster_> second_best_cluster; // i.e., ic2
    std::vector<Index_> cluster_sizes; // i.e., nc

    std::vector<Float_> loss_multiplier; // i.e., an1
    std::vector<Float_> gain_multiplier; // i.e., an2
    std::vector<Float_> wcss_loss; // i.e., d

    std::vector<UpdateHistory<Index_> > update_history; // i.e., ncp
    std::vector<uint8_t> was_quick_transferred; // i.e., itran
    std::vector<LiveStatus<Index_> > live_set; // i.e., live

    Index_ optra_steps_since_last_transfer = 0; // i.e., indx

public:
    Workspace(Index_ nobs, Cluster_ ncenters) :
        // Sizes taken from the .Fortran() call in stats::kmeans(). 
        second_best_cluster(nobs), 
        cluster_sizes(ncenters),
        loss_multiplier(ncenters),
        gain_multiplier(ncenters),
        wcss_loss(nobs),

        // All the other bits and pieces.
        update_history(ncenters),
        was_quick_transferred(ncenters),
        live_set(ncenters)
    {}
};

template<typename Data_, typename Float_, typename Dim_>
Float_ squared_distance_from_cluster(const Data_* data, const Float_* center, Dim_ ndim) {
    Float_ output = 0;
    for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++data, ++center) {
        Float_ delta = static_cast<Float_>(*data) - *center; // cast to float for consistent precision regardless of Data_.
        output += delta * delta;
    }
    return output;
}

template<class Matrix_, typename Cluster_, typename Float_>
void find_closest_two_centers(const Matrix_& data, Cluster_ ncenters, const Float_* centers, Cluster_* best_cluster, std::vector<Cluster_>& second_best_cluster, int nthreads) {
    auto ndim = data.num_dimensions();

    // We assume that there are at least two centers here, otherwise we should
    // have detected that this was an edge case in RefineHartiganWong::run.
    internal::QuickSearch<Float_, Cluster_, decltype(ndim)> index(ndim, ncenters, centers);

    auto nobs = data.num_observations();
    typedef typename Matrix_::index_type Index_;
    parallelize(nthreads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto matwork = data.create_workspace(start, length);
        for (Index_ obs = start, end = start + length; obs < end; ++obs) {
            auto optr = data.get_observation(matwork);
            auto res2 = index.find2(optr);
            best_cluster[obs] = res2.first;
            second_best_cluster[obs] = res2.second;
        }
    });
}

template<typename Float_>
constexpr Float_ big_number() {
    return 1e30; // Some very big number.
}

template<typename Dim_, typename Data_, typename Index_, typename Cluster_, typename Float_>
void transfer_point(Dim_ ndim, const Data_* obs_ptr, Index_ obs_id, Cluster_ l1, Cluster_ l2, Float_* centers, Cluster_* best_cluster, Workspace<Float_, Index_, Cluster_>& work) {
    // Yes, casts to float are deliberate here, so that the
    // multipliers can be computed correctly.
    Float_ al1 = work.cluster_sizes[l1], alw = al1 - 1;
    Float_ al2 = work.cluster_sizes[l2], alt = al2 + 1;

    size_t long_ndim = ndim;
    auto copy1 = centers + static_cast<size_t>(l1) * long_ndim; // cast to avoid overflow.
    auto copy2 = centers + static_cast<size_t>(l2) * long_ndim;
    for (decltype(ndim) dim = 0; dim < ndim; ++dim, ++copy1, ++copy2, ++obs_ptr) {
        Float_ oval = *obs_ptr; // cast to float for consistent precision regardless of Data_.
        *copy1 = (*copy1 * al1 - oval) / alw;
        *copy2 = (*copy2 * al2 + oval) / alt;
    }

    --work.cluster_sizes[l1];
    ++work.cluster_sizes[l2];

    work.gain_multiplier[l1] = alw / al1;
    work.loss_multiplier[l1] = (alw > 1 ? alw / (alw - 1) : big_number<Float_>());
    work.loss_multiplier[l2] = alt / al2;
    work.gain_multiplier[l2] = alt / (alt + 1);

    best_cluster[obs_id] = l2;
    work.second_best_cluster[obs_id] = l1;
}

/* ALGORITHM AS 136.1  APPL. STATIST. (1979) VOL.28, NO.1
 * This is the OPtimal TRAnsfer stage.
 *             ----------------------
 * Each point is re-assigned, if necessary, to the cluster that will induce a
 * maximum reduction in the within-cluster sum of squares. In this stage,
 * there is only one pass through the data.
 */
template<class Matrix_, typename Cluster_, typename Float_>
bool optimal_transfer(const Matrix_& data, Workspace<Float_, typename Matrix_::index_type, Cluster_>& work, Cluster_ ncenters, Float_* centers, Cluster_* best_cluster) {
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
            // earlier in the current optimal_transfer call. No need to worry
            // about quick_transfer as all WCSS losses are guaranteed to be
            // accurate when we exit from that function.
            auto& wcss_loss = work.wcss_loss[obs];
            auto& history1 = work.update_history[l1];
            if (!history1.is_unchanged()) {
                auto l1_ptr = centers + long_ndim * static_cast<size_t>(l1); // cast to avoid overflow.
                wcss_loss = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
            }

            // Find the cluster with minimum WCSS gain.
            auto l2 = work.second_best_cluster[obs];
            auto original_l2 = l2;
            auto l2_ptr = centers + long_ndim * static_cast<size_t>(l2); // cast to avoid overflow.
            auto wcss_gain = squared_distance_from_cluster(obs_ptr, l2_ptr, ndim) * work.gain_multiplier[l2];

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
            auto& live1 = work.live_set[l1];
            if (live1.is_live(obs)) { 
                for (Cluster_ cen = 0; cen < ncenters; ++cen) {
                    if (cen != l1 && cen != original_l2) {
                        check_best_cluster(cen);
                    }
                }
            } else {
                for (Cluster_ cen = 0; cen < ncenters; ++cen) {
                    if (cen != l1 && cen != original_l2 && work.live_set[cen].is_live(obs)) {
                        check_best_cluster(cen);
                    }
                }
            }

            // Deciding whether to make the transfer based on the change to the WCSS.
            if (wcss_gain >= wcss_loss) {
                work.second_best_cluster[obs] = l2;
            } else {
                work.optra_steps_since_last_transfer = 0;

                live1.mark_current(obs);
                work.live_set[l2].mark_current(obs);
                history1.set_optimal(obs);
                work.update_history[l2].set_optimal(obs);

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
template<class Matrix_, typename Cluster_, typename Float_>
std::pair<bool, bool> quick_transfer(
    const Matrix_& data,
    Workspace<Float_, typename Matrix_::index_type, Cluster_>& work,
    Float_* centers,
    Cluster_* best_cluster,
    int quick_iterations,
    bool quit_on_limit)
{
    bool had_transfer = false;
    std::fill(work.was_quick_transferred.begin(), work.was_quick_transferred.end(), 0);

    auto nobs = data.num_observations();
    auto matwork = data.create_workspace();
    auto ndim = data.num_dimensions();
    size_t long_ndim = data.num_dimensions();

    typedef decltype(nobs) Index_;
    Index_ steps_since_last_quick_transfer = 0;

    for (int it = 0; it < quick_iterations; ++it) {
        int prev_it = it - 1;

        for (decltype(nobs) obs = 0; obs < nobs; ++obs) { 
            ++steps_since_last_quick_transfer;
            auto l1 = best_cluster[obs];

            if (work.cluster_sizes[l1] != 1) {
                const typename Matrix_::data_type* obs_ptr = NULL;

                // Need to update the WCSS loss if the cluster was updated recently. 
                // Otherwise, we must have already updated the WCSS in a previous 
                // iteration of the outermost loop, so this can be skipped.
                //
                // Note that we use changed_at_or_after; if the same
                // observation was changed in the previous iteration of the
                // outermost loop, its WCSS loss won't have been updated yet.
                auto& history1 = work.update_history[l1];
                if (history1.changed_after_or_at(prev_it, obs)) {
                    auto l1_ptr = centers + static_cast<size_t>(l1) * long_ndim; // cast to avoid overflow.
                    obs_ptr = data.get_observation(obs, matwork);
                    work.wcss_loss[obs] = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
                }

                // If neither the best or second-best clusters have changed
                // after the previous iteration that we visited this
                // observation, then there's no point reevaluating the
                // transfer, because nothing's going to be different anyway.
                auto l2 = work.second_best_cluster[obs];
                auto& history2 = work.update_history[l2];
                if (history1.changed_after(prev_it, obs) || history2.changed_after(prev_it, obs)) {
                    if (obs_ptr == NULL) {
                        obs_ptr = data.get_observation(obs, matwork);
                    }
                    auto l2_ptr = centers + static_cast<size_t>(l2) * long_ndim; // cast to avoid overflow.
                    auto wcss_gain = squared_distance_from_cluster(obs_ptr, l2_ptr, ndim) * work.gain_multiplier[l2];

                    if (wcss_gain < work.wcss_loss[obs]) {
                        had_transfer = true;
                        steps_since_last_quick_transfer = 0;

                        work.was_quick_transferred[l1] = true;
                        work.was_quick_transferred[l2] = true;

                        history1.set_quick(it, obs);
                        history2.set_quick(it, obs);

                        transfer_point(ndim, obs_ptr, obs, l1, l2, centers, best_cluster, work);
                    }
               }
           }

           if (steps_since_last_quick_transfer == nobs) {
               // Quit early if no transfer occurred within the past 'nobs'
               // steps, as we've already converged for each observation. 
               return std::make_pair(had_transfer, false);
           }
        }
    } 

    if (!quit_on_limit) {
        // Making sure that the WCSS losses are correctly calculated, as the
        // last pass through the data may have done a transfer that would not
        // have triggered updates for points <= the updated point.
        int prev_it = quick_iterations - 1;
        for (decltype(nobs) obs = 0; obs < nobs; ++obs) { 
            auto l1 = best_cluster[obs];
            if (work.cluster_sizes[l1] != 1 && work.update_history[l1].changed_after_or_at(prev_it, obs)) {
                auto l1_ptr = centers + static_cast<size_t>(l1) * long_ndim; // cast to avoid overflow.
                auto obs_ptr = data.get_observation(obs, matwork);
                work.wcss_loss[obs] = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
            }
        }
    }

    return std::make_pair(had_transfer, true);
}

}
/**
 * @endcond
 */

/**
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 *
 * The Hartigan-Wong algorithm performs several iterations of transferring points between clusters, 
 * involving a computationally expensive "optimal transfer" step that checks each observation against each cluster to determine its best assignment,
 * followed by a cheaper "quick transfer" step, which iterates between the best and second-best cluster choices for each observation.
 * The latter enables rapid exploration of the local solution space without the unnecessary expense of repeatedly comparing each observation to all clusters.
 * The choice of "best" cluster for each observation considers the gain/loss in the sum of squares when an observation moves between clusters,
 * even accounting for the shift in the cluster centers after the transfer.
 * The algorithm terminates when no observation wishes to transfer between clusters.
 *
 * This implementation is derived from the Fortran code underlying the `kmeans` function in the **stats** R package,
 * which in turn is derived from Hartigan and Wong (1979).
 * 
 * In the `Details::status` returned by `run()`, the status code is either 0 (success),
 * 2 (maximum optimal transfer iterations reached without convergence)
 * or 4 (maximum quick transfer iterations reached without convergence, if `RefineHartiganWongOptions::quit_on_quick_transfer_convergence_failure = false`).
 * Previous versions of the library would report a status code of 1 upon encountering an empty cluster, but these are now just ignored.
 * 
 * @tparam Matrix_ Matrix type for the input data.
 * This should satisfy the `MockMatrix` contract.
 * @tparam Cluster_ Integer type for the cluster assignments.
 * @tparam Float_ Floating-point type for the centroids.
 *
 * @see
 * Hartigan, J. A. and Wong, M. A. (1979).
 * Algorithm AS 136: A K-means clustering algorithm.
 * _Applied Statistics_, 28, 100-108.
 */
template<typename Matrix_ = SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
class RefineHartiganWong : public Refine<Matrix_, Cluster_, Float_> {
public:
    /**
     * @param options Further options for the Hartigan-Wong algorithm.
     */
    RefineHartiganWong(RefineHartiganWongOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor. 
     */
    RefineHartiganWong() = default;

private:
    RefineHartiganWongOptions my_options;
    typedef typename Matrix_::index_type Index_;

public:
    /**
     * @return Options for Hartigan-Wong clustering,
     * to be modified prior to calling `run()`.
     */
    RefineHartiganWongOptions& get_options() {
        return my_options;
    }

public:
    Details<Index_> run(const Matrix_& data, Cluster_ ncenters, Float_* centers, Cluster_* clusters) const {
        auto nobs = data.num_observations();
        if (internal::is_edge_case(nobs, ncenters)) {
            return internal::process_edge_case(data, ncenters, centers, clusters);
        }

        RefineHartiganWong_internal::Workspace<Float_, Index_, Cluster_> work(nobs, ncenters);

        RefineHartiganWong_internal::find_closest_two_centers(data, ncenters, centers, clusters, work.second_best_cluster, my_options.num_threads);
        for (Index_ obs = 0; obs < nobs; ++obs) {
            ++work.cluster_sizes[clusters[obs]];
        }
        internal::compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);

        for (Cluster_ cen = 0; cen < ncenters; ++cen) {
            Float_ num = work.cluster_sizes[cen]; // yes, cast is deliberate here so that the multipliers can be computed correctly.
            work.gain_multiplier[cen] = num / (num + 1);
            work.loss_multiplier[cen] = (num > 1 ? num / (num - 1) : RefineHartiganWong_internal::big_number<Float_>());
        }

        int iter = 0;
        int ifault = 0;
        while ((++iter) <= my_options.max_iterations) {
            bool finished = RefineHartiganWong_internal::optimal_transfer(data, work, ncenters, centers, clusters);
            if (finished) {
                break;
            }

            auto quick_status = RefineHartiganWong_internal::quick_transfer(
                data,
                work,
                centers,
                clusters,
                my_options.max_quick_transfer_iterations,
                my_options.quit_on_quick_transfer_convergence_failure
            );

            if (quick_status.second) { // Hit the quick transfer iteration limit.
                if (my_options.quit_on_quick_transfer_convergence_failure) {
                    ifault = 4;
                    break;
                }
            } else {
                // If quick transfer converged and there are only two clusters,
                // there is no need to re-enter the optimal transfer stage. 
                if (ncenters == 2) {
                    break;
                }
            }

            if (quick_status.first) { // At least one transfer was performed.
                work.optra_steps_since_last_transfer = 0;
            }

            // If we get to this point, all WCSS losses are up to date, as
            // quick_transfer() updated everything for us; so we can set the
            // status to 'unchanged' for all clusters.
            for (auto& u : work.update_history) {
                u.set_unchanged();
            }

            for (Cluster_ c = 0; c < ncenters; ++c) {
                work.live_set[c].reset(work.was_quick_transferred[c]);
            }
        }

        if (iter == my_options.max_iterations + 1) {
            ifault = 2;
        }

        internal::compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);
        return Details(std::move(work.cluster_sizes), iter, ifault);
    }
};

}

#endif
