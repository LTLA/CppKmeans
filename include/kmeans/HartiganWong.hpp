#ifndef KMEANS_HARTIGAN_WONG_HPP
#define KMEANS_HARTIGAN_WONG_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <iostream>

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
 * @see
 * Hartigan, J. A. and Wong, M. A. (1979).
 * Algorithm AS 136: A K-means clustering algorithm.
 * _Applied Statistics_, 28, 100-108.
 */
class HartiganWong {
    int num_dim, num_obs;
    const double* data_ptr;

    int num_centers;
    double* centers_ptr;

    // Subsequent arguments in the same order as supplied to R's kmns_ function.
    int * ic1;
    std::vector<int> ic2, nc;
    std::vector<double> an1, an2;
    std::vector<int> ncp;
    std::vector<double> d;

    std::vector<uint8_t> itran;
    std::vector<int> live;

    static constexpr double big = 1e30; // Define BIG to be a very large positive number
    static constexpr int ncp_init = -2;
    static constexpr int ncp_unchanged = -1;
public:
    /**
     * @brief Statistics from the Hartigan-Wong algorithm.
     */
    struct Results {
        /**
         * @cond
         */
        Results(std::vector<int> s, std::vector<double> w, int it, int st) : sizes(std::move(s)), withinss(std::move(w)), iterations(it), status(st) {} 
        /**
         * @endcond
         */

        /**
         * The number of observations in each cluster.
         * All values are guaranteed to be positive for non-zero numbers of observations unless `status == 1` or `3`.
         */
        std::vector<int> sizes;

        /**
         * The within-cluster sum of squares for each cluster.
         * All values are guaranteed to be non-negative.
         * Values may be zero for clusters with only one observation.
         */
        std::vector<double> withinss;

        /**
         * The number of iterations used to achieve convergence.
         * This value may be greater than the `maxit` if convergence was not achieved.
         */
        int iterations;

        /**
         * The status of the algorithm.
         *
         * - 0: convergence achieved.
         * - 1: empty cluster detected.
         * This usually indicates a problem with the initial choice of centroids.
         * It can also occur in pathological situations with duplicate points.
         * - 2: maximum iterations reached without convergence. 
         * - 3: the number of centers is not positive or greater than the number of observations.
         * - 4: maximum number of quick transfer steps exceeded.
         */
        int status;
    };

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] data Pointer to a `ndim`-by-`nobs` array where columns are observations and rows are dimensions. 
     * Data should be stored in column-major order.
     * @param ncenters Number of cluster centers.
     * @param[in, out] centers Pointer to a `ndim`-by-`ncenters` array where columns are cluster centers and rows are dimensions. 
     * On input, this should contain the initial centroid locations for each cluster.
     * Data should be stored in column-major order.
     * On output, this will contain the final centroid locations for each cluster.
     * @param[out] clusters Pointer to an array of length `nobs`.
     * Om output, this will contain the cluster assignment for each observation.
     * @param maxit Maximum number of iterations.
     *
     * @return `centers` and `clusters` are filled, and a `Results` object is returned containing clustering statistics.
     */
    Results run(int ndim, int nobs, const double* data, int ncenters, double* centers, int* clusters, int maxit = 10) {
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

        int iter, ifault;
        auto wcss = kmeans(maxit, iter, ifault);
        return Results(std::move(nc), std::move(wcss), iter, ifault);
    } 

private:
    // Returns the WCSS, modifies 'iter' and 'ifault'.
    std::vector<double> kmeans(int maxiter, int& iter, int& ifault) {
        iter = 0;
        ifault = 0;

        if (num_centers == 1) {
            // All points in cluster 0.
            std::fill(ic1, ic1 + num_obs, 0);
            nc[0] = num_obs;
            return compute_wcss();

        } else if (num_centers >= num_obs) {
            // Special case, each observation is a center.
            std::iota(ic1, ic1 + num_obs, 0);            
            std::fill(nc.begin(), nc.begin() + num_obs, 1);
            if (num_centers > num_obs) {
                std::fill(nc.begin() + num_obs, nc.end(), 0);
                ifault = 3;
            }
            return compute_wcss();

        } else if (num_centers == 0) {
            // No need to fill 'nc', it's already all-zero on input.
            ifault = 3;
            std::fill(ic1, ic1 + num_obs, 0);
            return std::vector<double>();
        }

        /* For each point I, find its two closest centres, IC1(I) and 
         * IC2(I). Assign it to IC1(I). 
         */
        for (int obs = 0; obs < num_obs; ++obs) {
            auto& best = ic1[obs];
            best = 0;
            double best_dist = squared_distance_from_cluster(obs, best);

            auto& second = ic2[obs];
            second = 1;
            double second_dist = squared_distance_from_cluster(obs, second);

            if (best_dist > second_dist) {
                std::swap(best, second);
                std::swap(best_dist, second_dist);
            }

            for (int cen = 2; cen < num_centers; ++cen) {
                double candidate_dist = squared_distance_from_cluster(obs, cen);
                if (candidate_dist < best_dist) {
                    std::swap(best_dist, second_dist);
                    std::swap(best, second);
                    best_dist = candidate_dist;
                    best = cen;
                } else if (candidate_dist < second_dist) {
                    second_dist = candidate_dist;
                    second = cen;                    
                }
            }
        }

        /* Update cluster centres to be the average of points contained 
         * within them. 
         * NC(L) := #{units in cluster L},  L = 1..K 
         */
        std::fill(centers_ptr, centers_ptr + num_dim * num_centers, 0);
        std::fill(nc.begin(), nc.end(), 0);
        for (int obs = 0; obs < num_obs; ++obs) {
            auto cen = ic1[obs];
            ++nc[cen];
            add_point_to_cluster(obs, cen);
        }

        // Check to see if there is any empty cluster at this stage 
        for (int cen = 0; cen < num_centers; ++cen) {
            if (nc[cen] == 0) {
                ifault = 1;
                return compute_wcss();
            }

            divide_by_cluster_size(cen);

            /* Initialize AN1, AN2.
             * AN1(L) = NC(L) / (NC(L) - 1)
             * AN2(L) = NC(L) / (NC(L) + 1)
             */
            const double num = nc[cen];
            an2[cen] = num / (num + 1);
            an1[cen] = (num > 1 ? num / (num - 1) : big);
        }

        int indx = 0;
        int imaxqtr = num_obs * 50; // default derived from stats::kmeans()
        std::fill(ncp.begin(), ncp.end(), ncp_init);
        std::fill(itran.begin(), itran.end(), true);
        std::fill(live.begin(), live.end(), 0);

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
            std::fill(ncp.begin(), ncp.end(), ncp_unchanged);
        }

        /* Since the specified number of iterations has been exceeded, set
         * IFAULT = 2. This may indicate unforeseen looping.
         */
        if (iter == maxiter + 1) {
            ifault = 2;
        }

        return compute_wcss();
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

    void add_point_to_cluster(int obs, int cen) {
        auto copy = centers_ptr + cen * num_dim;
        auto mine = data_ptr + obs * num_dim;
        for (int dim = 0; dim < num_dim; ++dim, ++copy, ++mine) {
            *copy += *mine;
        }
    }

    void divide_by_cluster_size(int cen) {
        if (nc[cen]) {
            auto curcenter = centers_ptr + cen * num_dim;
            for (int dim = 0; dim < num_dim; ++dim, ++curcenter) {
                *curcenter /= nc[cen];
            }
        }
    }

    std::vector<double> compute_wcss() {
        std::fill(centers_ptr, centers_ptr + num_centers * num_dim, 0);
        std::vector<double> wcss(num_centers);
        std::fill(wcss.begin(), wcss.end(), 0);

        for (int obs = 0; obs < num_obs; ++obs) {
            add_point_to_cluster(obs, ic1[obs]);
        }
        for (int cen = 0; cen < num_centers; ++cen) {
            divide_by_cluster_size(cen);
        }

        for (int obs = 0; obs < num_obs; ++obs) {
            auto cen = ic1[obs];
            auto curcenter = centers_ptr + cen * num_dim;
            auto& curwcss = wcss[cen];

            auto curdata = data_ptr + obs * num_dim;
            for (int dim = 0; dim < num_dim; ++dim, ++curcenter, ++curdata) {
                curwcss += (*curdata - *curcenter) * (*curdata - *curcenter);
            }
        }

        return wcss;
    }

    double squared_distance_from_cluster(int pt, int clust) const {
        const double* acopy = data_ptr + pt * num_dim;
        const double* ccopy = centers_ptr + clust * num_dim;
        double output = 0;
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
    void optimal_transfer(int& indx) {
        /* If cluster L is updated in the last quick-transfer stage, it 
         * belongs to the live set throughout this stage. Otherwise, at 
         * each step, it is not in the live set if it has not been updated 
         * in the last M optimal transfer steps. (AL: M being a synonym for
         * the number of observations, here and in other functions.)
         */
        for (int cen = 0; cen < num_centers; ++cen) {
            if (itran[cen]) {
                live[cen] = num_obs; // AL: using 0-index, so no need for +1.
            }
        }

        for (int obs = 0; obs < num_obs; ++obs) { 
            ++indx;
            auto l1 = ic1[obs];

            // If point I is the only member of cluster L1, no transfer.
            if (nc[l1] != 1) {
                // If L1 has not yet been updated in this stage, no need to re-compute D(I).
                if (ncp[l1] != ncp_unchanged) {
                    d[obs] = squared_distance_from_cluster(obs, l1) * an1[l1];
                }

                // Find the cluster with minimum R2.
                auto l2 = ic2[obs];
                auto ll = l2;
                double r2 = squared_distance_from_cluster(obs, l2) * an2[l2];
            
                for (int cen = 0; cen < num_centers; ++cen) {
                    /* If I >= LIVE(L1), then L1 is not in the live set. If this is
                     * true, we only need to consider clusters that are in the live
                     * set for possible transfer of point I. Otherwise, we need to
                     * consider all possible clusters. 
                     */
                    if (obs >= live[l1] && obs >= live[cen] || cen == l1 || cen == ll) {
                        continue;
                    }

                    double rr = r2 / an2[cen];
                    double dc = squared_distance_from_cluster(obs, cen);
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
                    ncp[l1] = obs;
                    ncp[l2] = obs;

                    transfer_point(obs, l1, l2);
                }
            }

            if (indx == num_obs) {
                return;
            }
        }

        for (int cen = 0; cen < num_centers; ++cen) {
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
    void quick_transfer (int& indx, int& imaxqtr) {
        int icoun = 0;
        int istep = 0;

        while (1) {
           for (int obs = 0; obs < num_obs; ++obs) { 
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
                   if (istep <= ncp[l1]) {
                       d[obs] = squared_distance_from_cluster(obs, l1) * an1[l1];
                   }

                   // If ISTEP >= both NCP(L1) & NCP(L2) there will be no transfer of point I at this step. 
                   auto l2 = ic2[obs];
                   if (istep < ncp[l1] || istep < ncp[l2]) {
                       if (squared_distance_from_cluster(obs, l2) < d[obs] / an2[l2]) {
                           /* Update cluster centres, NCP, NC, ITRAN, AN1 & AN2 for clusters
                            * L1 & L2.   Also update IC1(I) & IC2(I).   Note that if any
                            * updating occurs in this stage, INDX is set back to 0. 
                            */
                           icoun = 0;
                           indx = 0;

                           itran[l1] = true;
                           itran[l2] = true;
                           ncp[l1] = istep + num_obs;
                           ncp[l2] = istep + num_obs;
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
    void transfer_point(int obs, int l1, int l2) {
        const double al1 = nc[l1], alw = al1 - 1;
        const double al2 = nc[l2], alt = al2 + 1;

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
