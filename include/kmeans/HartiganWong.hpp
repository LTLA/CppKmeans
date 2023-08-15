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
template<typename DATA_t = double, typename CLUSTER_t = int, typename INDEX_t = int>
class HartiganWong : public Refine<DATA_t, CLUSTER_t, INDEX_t> {
    int num_dim;
    INDEX_t num_obs;
    const DATA_t* data_ptr;

    CLUSTER_t num_centers;
    DATA_t* centers_ptr;

    // Array arguments in the same order as supplied to R's kmns_ function.
    CLUSTER_t * ic1;
    std::vector<CLUSTER_t> ic2;
    std::vector<INDEX_t> nc;
    std::vector<DATA_t> an1, an2;
    std::vector<INDEX_t> ncp;
    std::vector<DATA_t> d;
    std::vector<uint8_t> itran;
    std::vector<INDEX_t> live;

private:
    static constexpr double big = 1e30; // Define BIG to be a very large positive number

    /* The following comparisons to 'ncp' are wrapped in functions to account
     * for the fact that we need to shift all the 'ncp' values by two to give
     * some space for the error codes when dealing with unsigned integers in
     * 'INDEX_t'. All interactions with 'ncp' should occur via these utilities.
     */
    static constexpr INDEX_t ncp_init = 0;
    static constexpr INDEX_t ncp_unchanged = 1;
    static constexpr INDEX_t ncp_shift = 2;

    void initialize_ncp() {
        std::fill(ncp.begin(), ncp.end(), ncp_init);
    }

    void reset_ncp() {
        std::fill(ncp.begin(), ncp.end(), ncp_unchanged);
    }
       
    void set_ncp(INDEX_t obs, INDEX_t val) {
        ncp[obs] = val + ncp_shift;
    }

    bool unchanged_ncp(INDEX_t obs) const {
        return ncp[obs] == ncp_unchanged;
    }

    bool lt_ncp(INDEX_t obs, INDEX_t val) const {
        return ncp[obs] > val + ncp_shift;
    }

    bool le_ncp(INDEX_t obs, INDEX_t val) const {
        return ncp[obs] >= val + ncp_shift;
    }

public:
    /** 
     * @brief Default parameter values for `HartiganWong`.
     */
    struct Defaults {
        /** 
         * See `set_max_iterations()` for more details.
         */
        static constexpr int max_iterations = 10;

        /** 
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };

private:
    int maxiter = Defaults::max_iterations;
    int nthreads = Defaults::num_threads;

public:
    /**
     * @param m Maximum number of iterations.
     * More iterations increase the opportunity for convergence at the cost of more computational time.
     *
     * @return A reference to this `HartiganWong` object.
     */
    HartiganWong& set_max_iterations(int m = Defaults::max_iterations) {
        maxiter = m;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `HartiganWong` object.
     */
    HartiganWong& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

public:
    Details<DATA_t, INDEX_t> run(int ndim, INDEX_t nobs, const DATA_t* data, CLUSTER_t ncenters, DATA_t* centers, CLUSTER_t* clusters) {
        num_dim = ndim;
        num_obs = nobs;
        data_ptr = data;
        num_centers = ncenters;
        centers_ptr = centers; 
        ic1 = clusters;

        // Sizes taken from the .Fortran() call in stats::kmeans(). 
        ic2.resize(num_obs);
        nc.resize(num_centers);
        an1.resize(num_centers);
        an2.resize(num_centers);
        d.resize(num_obs);

        /* ITRAN(L) = 1 if cluster L is updated in the quick-transfer stage,
         *          = 0 otherwise
         * In the optimal-transfer stage, NCP(L) stores the step at which
         * cluster L is last updated.
         * In the quick-transfer stage, NCP(L) stores the step at which
         * cluster L is last updated plus M. 
         */
        ncp.resize(num_centers);
        itran.resize(num_centers);
        live.resize(num_centers);

        if (is_edge_case(num_obs, num_centers)) {
            return process_edge_case(num_dim, num_obs, data_ptr, num_centers, centers_ptr, ic1);
        }

        /* For each point I, find its two closest centres, IC1(I) and 
         * IC2(I). Assign it to IC1(I). 
         */
#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
#endif
        for (INDEX_t obs = 0; obs < num_obs; ++obs) {
#else
        KMEANS_CUSTOM_PARALLEL(num_obs, [&](INDEX_t first, INDEX_t last) -> void {
        for (INDEX_t obs = first; obs < last; ++obs) {
#endif
            auto& best = ic1[obs];
            best = 0;
            DATA_t best_dist = squared_distance_from_cluster(obs, best);

            auto& second = ic2[obs];
            second = 1;
            DATA_t second_dist = squared_distance_from_cluster(obs, second);

            if (best_dist > second_dist) {
                std::swap(best, second);
                std::swap(best_dist, second_dist);
            }

            for (CLUSTER_t cen = 2; cen < num_centers; ++cen) {
                DATA_t candidate_dist = squared_distance_from_cluster(obs, cen);
                if (candidate_dist < second_dist) {
                    second_dist = candidate_dist;
                    second = cen;                    
                    if (candidate_dist < best_dist) {
                        std::swap(best_dist, second_dist);
                        std::swap(best, second);
                    }
                }
            }
#ifndef KMEANS_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        /* Update cluster centres to be the average of points contained 
         * within them. 
         * NC(L) := #{units in cluster L},  L = 1..K 
         */
        std::fill(nc.begin(), nc.end(), 0);
        for (INDEX_t obs = 0; obs < num_obs; ++obs) {
            ++nc[ic1[obs]];
        }

        compute_centroids(num_dim, num_obs, data_ptr, num_centers, centers_ptr, ic1, nc);

        // Check to see if there is any empty cluster at this stage 
        for (CLUSTER_t cen = 0; cen < num_centers; ++cen) {
            if (nc[cen] == 0) {
                return Details<DATA_t, INDEX_t>(0, 1); // i.e., ifault = 1 here.
            }

            /* Initialize AN1, AN2.
             * AN1(L) = NC(L) / (NC(L) - 1)
             * AN2(L) = NC(L) / (NC(L) + 1)
             */
            const DATA_t num = nc[cen];
            an2[cen] = num / (num + 1);
            an1[cen] = (num > 1 ? num / (num - 1) : big);
        }

        INDEX_t indx = 0;
        if (std::numeric_limits<INDEX_t>::max() / 50 < num_obs) {
            throw std::runtime_error("too many observations for the index integer type");
        }
        INDEX_t imaxqtr = num_obs * 50; // default derived from stats::kmeans()

        initialize_ncp();
        std::fill(itran.begin(), itran.end(), true);
        std::fill(live.begin(), live.end(), 0);
        int iter = 0;
        int ifault = 0;

        for (iter = 1; iter <= maxiter; ++iter) {

            /* OPtimal-TRAnsfer stage: there is only one pass through the data. 
             * Each point is re-allocated, if necessary, to the cluster that will
             * induce the maximum reduction in within-cluster sum of squares.
             */
            optimal_transfer(indx);

            // Stop if no transfer took place in the last M optimal transfer steps.
            if (indx == num_obs) {
                break;
            }

            /* Quick-TRANSfer stage: Each point is tested in turn to see if it should
             * be re-allocated to the cluster to which it is most likely to be
             * transferred, IC2(I), from its present cluster, IC1(I). 
             * Loop through the data until no further change is to take place. 
             */
            quick_transfer(indx, imaxqtr);

            if (imaxqtr < 0) {
                ifault = 4;
                break;
            }

            // If there are only two clusters, there is no need to re-enter the optimal transfer stage. 
            if (num_centers == 2) {
                break;
            }

            // NCP has to be reset before entering optimal_transfer().
            reset_ncp();
        }

        /* Since the specified number of iterations has been exceeded, set
         * IFAULT = 2. This may indicate unforeseen looping.
         */
        if (iter == maxiter + 1) {
            ifault = 2;
        }

        compute_centroids(num_dim, num_obs, data_ptr, num_centers, centers_ptr, ic1, nc);
        return Details(
            std::move(nc),
            compute_wcss(num_dim, num_obs, data_ptr, num_centers, centers_ptr, ic1),
            iter,
            ifault
        );
    }

private:
#ifdef DEBUG
    template<class T>
    void print_vector(const T& vec, const char* msg) {
        std::cout << msg << std::endl;
        for (auto c : vec) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
#endif

    DATA_t squared_distance_from_cluster(INDEX_t pt, CLUSTER_t clust) const {
        const DATA_t* acopy = data_ptr + pt * num_dim;
        const DATA_t* ccopy = centers_ptr + clust * num_dim;
        DATA_t output = 0;
        for (int dim = 0; dim < num_dim; ++dim, ++acopy, ++ccopy) {
            output += (*acopy - *ccopy) * (*acopy - *ccopy);
        }
        return output;
    }

private:
    /* ALGORITHM AS 136.1  APPL. STATIST. (1979) VOL.28, NO.1
     * This is the OPtimal TRAnsfer stage.
     *             ----------------------
     * Each point is re-allocated, if necessary, to the cluster that
     * will induce a maximum reduction in the within-cluster sum of
     * squares. 
     */
    void optimal_transfer(INDEX_t& indx) {
        /* If cluster L is updated in the last quick-transfer stage, it 
         * belongs to the live set throughout this stage. Otherwise, at 
         * each step, it is not in the live set if it has not been updated 
         * in the last M optimal transfer steps. (AL: M being a synonym for
         * the number of observations, here and in other functions.)
         */
        for (CLUSTER_t cen = 0; cen < num_centers; ++cen) {
            if (itran[cen]) {
                live[cen] = num_obs; // AL: using 0-index, so no need for +1.
            }
        }

        for (INDEX_t obs = 0; obs < num_obs; ++obs) { 
            ++indx;
            auto l1 = ic1[obs];

            // If point I is the only member of cluster L1, no transfer.
            if (nc[l1] != 1) {
                // If L1 has not yet been updated in this stage, no need to re-compute D(I).
                if (!unchanged_ncp(l1)) {
                    d[obs] = squared_distance_from_cluster(obs, l1) * an1[l1];
                }

                // Find the cluster with minimum R2.
                auto l2 = ic2[obs];
                auto ll = l2;
                DATA_t r2 = squared_distance_from_cluster(obs, l2) * an2[l2];
            
                for (CLUSTER_t cen = 0; cen < num_centers; ++cen) {
                    /* If I >= LIVE(L1), then L1 is not in the live set. If this is
                     * true, we only need to consider clusters that are in the live
                     * set for possible transfer of point I. Otherwise, we need to
                     * consider all possible clusters. 
                     *
                     * Translation note: the original Fortran code doesn't have parentheses
                     * around the first &&, but .AND. has greater precedence than .OR.,
                     * so it should be faithful to add those parentheses for clarity
                     * (and to avoid GCC warnings).
                     */
                    if ((obs >= live[l1] && obs >= live[cen]) || cen == l1 || cen == ll) {
                        continue;
                    }

                    DATA_t rr = r2 / an2[cen];
                    DATA_t dc = squared_distance_from_cluster(obs, cen);
                    if (dc < rr) {
                        r2 = dc * an2[cen];
                        l2 = cen;
                    }
                }

                if (r2 >= d[obs]) {
                    // If no transfer is necessary, L2 is the new IC2(I).
                    ic2[obs] = l2;

                } else {
                    /* Update cluster centres, LIVE, NCP, AN1 & AN2 for clusters L1 and 
                     * L2, and update IC1(I) & IC2(I). 
                     */
                    indx = 0;
                    live[l1] = num_obs + obs;
                    live[l2] = num_obs + obs;
                    set_ncp(l1, obs);
                    set_ncp(l2, obs);

                    transfer_point(obs, l1, l2);
                }
            }

            if (indx == num_obs) {
                return;
            }
        }

        for (CLUSTER_t cen = 0; cen < num_centers; ++cen) {
            itran[cen] = false;

            // LIVE(L) has to be decreased by M before re-entering OPTRA.
            // This means that if I >= LIVE(L1) in the next OPTRA call,
            // the last update must be >= M steps ago, as we effectively
            // 'lapped' the previous update for this cluster.
            live[cen] -= num_obs;
        }

        return;
    } 

private:
    /*     ALGORITHM AS 136.2  APPL. STATIST. (1979) VOL.28, NO.1 
     *     This is the Quick TRANsfer stage. 
     *                 -------------------- 
     *     IC1(I) is the cluster which point I belongs to.
     *     IC2(I) is the cluster which point I is most likely to be 
     *         transferred to.
     *
     *     For each point I, IC1(I) & IC2(I) are switched, if necessary, to 
     *     reduce within-cluster sum of squares.  The cluster centres are 
     *     updated after each step. 
     */
    void quick_transfer (INDEX_t& indx, INDEX_t& imaxqtr) {
        INDEX_t icoun = 0;
        INDEX_t istep = 0;

        while (1) {
           for (INDEX_t obs = 0; obs < num_obs; ++obs) { 
               ++icoun;
               auto l1 = ic1[obs];

               // point I is the only member of cluster L1, no transfer.
               if (nc[l1] != 1) {

                   /* NCP(L) is equal to the step at which cluster L is last updated plus M.
                    * (AL: M is the notation for the number of observations, a.k.a. 'num_obs').
                    *
                    * If ISTEP > NCP(L1), no need to re-compute distance from point I to 
                    * cluster L1. Note that if cluster L1 is last updated exactly M 
                    * steps ago, we still need to compute the distance from point I to 
                    * cluster L1.
                    */
                   if (le_ncp(l1, istep)) {
                       d[obs] = squared_distance_from_cluster(obs, l1) * an1[l1];
                   }

                   // If ISTEP >= both NCP(L1) & NCP(L2) there will be no transfer of point I at this step. 
                   auto l2 = ic2[obs];
                   if (lt_ncp(l1, istep) || lt_ncp(l2, istep)) {
                       if (squared_distance_from_cluster(obs, l2) < d[obs] / an2[l2]) {
                           /* Update cluster centres, NCP, NC, ITRAN, AN1 & AN2 for clusters
                            * L1 & L2.   Also update IC1(I) & IC2(I).   Note that if any
                            * updating occurs in this stage, INDX is set back to 0. 
                            */
                           icoun = 0;
                           indx = 0;

                           itran[l1] = true;
                           itran[l2] = true;
                           set_ncp(l1, istep + num_obs);
                           set_ncp(l2, istep + num_obs);
                           transfer_point(obs, l1, l2);
                       }
                   }
               }

               // If no re-allocation took place in the last M steps, return.
               if (icoun == num_obs) {
                   return;
               }

               // AL: incrementing ISTEP after checks against NCP(L1), to avoid off-by-one 
               // errors after switching to zero-indexing for the observations.
               ++istep;
               if (istep >= imaxqtr) {
                   imaxqtr = -1;
                   return;
               }
            }
        } 
    }

private:
    void transfer_point(INDEX_t obs, CLUSTER_t l1, CLUSTER_t l2) {
        const DATA_t al1 = nc[l1], alw = al1 - 1;
        const DATA_t al2 = nc[l2], alt = al2 + 1;

        auto copy1 = centers_ptr + l1 * num_dim;
        auto copy2 = centers_ptr + l2 * num_dim;
        auto acopy = data_ptr + obs * num_dim;
        for (int dim = 0; dim < num_dim; ++dim, ++copy1, ++copy2, ++acopy) {
            *copy1 = (*copy1 * al1 - *acopy) / alw;
            *copy2 = (*copy2 * al2 + *acopy) / alt;
        }

        --nc[l1];
        ++nc[l2];

        an2[l1] = alw / al1;
        an1[l1] = (alw > 1 ? alw / (alw - 1) : big);
        an1[l2] = alt / al2;
        an2[l2] = alt / (alt + 1);

        ic1[obs] = l2;
        ic2[obs] = l1;
    }
};

}

#endif
