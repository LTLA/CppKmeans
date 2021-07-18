#ifndef KMEANS_HARTIGAN_WONG_HPP
#define KMEANS_HARTIGAN_WONG_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace kmeans {

class HartiganWong {
    const int ndim, nobs;
    const double* data;

    const int ncenters;
    double* centers;

    // Subsequent arguments in the same order as supplied to R's kmns_ function.
    std::vector<int> ic1, ic2, nc;
    std::vector<double> an1, an2;
    std::vector<int> ncp;
    std::vector<double> d;

    std::vector<uint8_t> itran;
    std::vector<int> live;
    const int maxiter;
    std::vector<double> wcss;

    int ifault = 0;
    int iter = 0;

    static constexpr double big = 1e30; // Define BIG to be a very large positive number
    static constexpr int ncp_init = -2;
    static constexpr int ncp_unchanged = -1;
public:
    HartiganWong(int nd, int no, const double* d, int nc, double* c, int maxit = 10) :
        ndim(nd), 
        nobs(no), 
        data(d), 
        ncenters(nc), 
        centers(c), 

        // Sizes taken from the .Fortran() call in stats::kmeans(). 
        ic1(nobs),
        ic2(nobs),
        nc(ncenters),
        an1(ncenters),
        an2(ncenters),
        d(nobs),

        /* ITRAN(L) = 1 if cluster L is updated in the quick-transfer stage,
         *          = 0 otherwise
         * In the optimal-transfer stage, NCP(L) stores the step at which
         * cluster L is last updated.
         * In the quick-transfer stage, NCP(L) stores the step at which
         * cluster L is last updated plus M. 
         */
        ncp(ncenters, ncp_init),
        itran(ncenters, true),

        live(ncenters),
        maxiter(maxit),
        wcss(ncenters)
    {
        kmeans();
    } 

    const std::vector<int>& clusters () const {
        return ic1;
    }

    const std::vector<int>& sizes() const {
        return nc;
    }

    const std::vector<double>& WCSS() const {
        return wcss;
    }

    const int iterations() const {
        return iter;
    }

    const int status() const {
        return ifault;
    }

private:
    void kmeans() {
        if (ncenters == 1) {
            // All points in cluster 0.
            nc[0] = nobs;
            compute_wcss();
            return;

        } else if (ncenters == nobs) {
            // Special case, each observation is a center.
            std::iota(ic1.begin(), ic1.end(), 0);
            std::fill(nc.begin(), nc.end(), 1);
            compute_wcss();
            return;

        } else if (ncenters > nobs) {
            throw std::runtime_error("number of centers must be less than number of observations"); 
        } else if (ncenters == 0) {
            throw std::runtime_error("number of centers must be positive");
        }

        /* For each point I, find its two closest centres, IC1(I) and 
         * IC2(I). Assign it to IC1(I). 
         */
        for (int obs = 0; obs < nobs; ++obs) {
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

            for (int cen = 2; cen < ncenters; ++cen) {
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
        std::fill(centers, centers + ndim * ncenters, 0);
        for (int obs = 0; obs < nobs; ++obs) {
            auto cen = ic1[obs];
            ++nc[cen];
            add_point_to_cluster(obs, cen);
        }

        // Check to see if there is any empty cluster at this stage 
        for (int cen = 0; cen < ncenters; ++cen) {
            if (nc[cen] == 0) {
                ifault = 1;
                return;
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
        int imaxqtr = nobs * 50; // default derived from stats::kmeans()

        for (iter = 1; iter <= maxiter; ++iter) {
            /* OPtimal-TRAnsfer stage: there is only one pass through the data. 
             * Each point is re-allocated, if necessary, to the cluster that will
             * induce the maximum reduction in within-cluster sum of squares.
             */
            optimal_transfer(indx);

            // Stop if no transfer took place in the last M optimal transfer steps.
            if (indx == nobs) {
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
            if (ncenters == 2) {
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

        compute_wcss();
    }

private:
    void add_point_to_cluster(int obs, int cen) {
        auto copy = centers + cen * ndim;
        auto mine = data + obs * ndim;
        for (int dim = 0; dim < ndim; ++dim, ++copy, ++mine) {
            *copy += *mine;
        }
    }

    void divide_by_cluster_size(int cen) {
        auto curcenter = centers + cen * ndim;
        for (int dim = 0; dim < ndim; ++dim, ++curcenter) {
            *curcenter /= nc[cen];
        }
    }

    void compute_wcss() {
        std::fill(centers, centers + ncenters * ndim, 0);
        std::fill(wcss.begin(), wcss.end(), 0);

        for (int obs = 0; obs < nobs; ++obs) {
            add_point_to_cluster(obs, ic1[obs]);
        }
        for (int cen = 0; cen < ncenters; ++cen) {
            divide_by_cluster_size(cen);
        }

        for (int obs = 0; obs < nobs; ++obs) {
            auto cen = ic1[obs];
            auto curcenter = centers + cen * ndim;
            auto& curwcss = wcss[cen];

            auto curdata = data + obs * ndim;
            for (int dim = 0; dim < ndim; ++dim, ++curcenter, ++curdata) {
                curwcss += (*curdata - *curcenter) * (*curdata - *curcenter);
            }
        }

        return;
    }

private:
    double squared_distance_from_cluster(int pt, int clust) const {
        const double* acopy = data + pt * ndim;
        const double* ccopy = centers + clust * ndim;
        double output = 0;
        for (int dim = 0; dim < ndim; ++dim, ++acopy, ++ccopy) {
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
        for (int cen = 0; cen < ncenters; ++cen) {
            if (itran[cen]) {
                live[cen] = nobs; // AL: using 0-index, so no need for +1.
            }
        }

        for (int obs = 0; obs < nobs; ++obs) { 
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
            
                for (int cen = 0; cen < ncenters; ++cen) {
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
                    live[l1] = nobs + obs;
                    live[l2] = nobs + obs;
                    ncp[l1] = obs;
                    ncp[l2] = obs;

                    transfer_point(obs, l1, l2);
                }
            }

            if (indx == nobs) {
                return;
            }
        }

        for (int cen = 0; cen < ncenters; ++cen) {
            itran[cen] = false;

            // LIVE(L) has to be decreased by M before re-entering OPTRA.
            // This means that if I >= LIVE(L1) in the next OPTRA call,
            // the last update must be >= M steps ago, given that we 
            // looped through all the observations already.
            live[cen] -= nobs;
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
           for (int obs = 0; obs < nobs; ++obs, ++icoun) { 
                auto l1 = ic1[obs];
                auto l2 = ic2[obs];

                // point I is the only member of cluster L1, no transfer.
                if (nc[l1] != 1) {

                    /* NCP(L) is equal to the step at which cluster L is last updated plus M.
                     * (AL: M is the notation for the number of observations, a.k.a. 'nobs').
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
                            ncp[l1] = istep + nobs;
                            ncp[l2] = istep + nobs;
                            transfer_point(obs, l1, l2);
                        }
                    }
                }

                // If no re-allocation took place in the last M steps, return.
                if (icoun == nobs) {
                    return;
                }

                // AL: incrementing after checks against NCP(L1), to avoid off-by-one 
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
        const double al2 = nc[l2], alt = al2 - 1;

        auto copy1 = centers + l1 * ndim;
        auto copy2 = centers + l2 * ndim;
        auto acopy = data + obs * ndim;
        for (int dim = 0; dim < ndim; ++dim, ++copy1, ++copy2, ++acopy) {
            *copy1 = (*copy1 * al1 - *acopy) / alw;
            *copy2 = (*copy2 * al2 - *acopy) / alt;
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
