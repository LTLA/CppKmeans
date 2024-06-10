#ifndef KMEANS_QUICKSEARCH_HPP
#define KMEANS_QUICKSEARCH_HPP

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <queue>
#include <cstdint>

namespace kmeans {

namespace internal {

/* Adapted from http://stevehanov.ca/blog/index.php?id=130 */
template<typename Float_, typename Index_, typename Dim_>
class QuickSearch {
private:
    Dim_ num_dim;
    size_t long_num_dim;

    template<typename Query_>
    static Float_ raw_distance(const Float_* x, const Query_* y, Dim_ ndim) {
        Float_ output = 0;
        for (Dim_ i = 0; i < ndim; ++i, ++x, ++y) {
            Float_ delta = *x - static_cast<Float_>(*y); // cast to ensure consistent precision regardless of Query_.
            output += delta * delta;
        }
        return output;
    }

private:
    typedef std::pair<Float_, Index_> DataPoint;
    std::vector<DataPoint> items;

private:
    // Index_ might be unsigned, so we use zero as the LEAF marker.
    static const Index_ LEAF = 0;

    struct Node {
        const Float_* center = NULL;
        Float_ radius = 0;

        // Original index of current vantage point
        Index_ index = 0;

        // Node index of the next vantage point for all children closer than 'threshold' from the current vantage point.
        // This must be > 0, as the first node in 'nodes' is the root and cannot be referenced from other nodes.
        Index_ left = LEAF;

        // Node index of the next vantage point for all children further than 'threshold' from the current vantage point.
        // This must be > 0, as the first node in 'nodes' is the root and cannot be referenced from other nodes.
        Index_ right = LEAF;
    };

    std::vector<Node> nodes;

private:
    template<class Engine_>
    Index_ build(Index_ lower, Index_ upper, const Float_* coords, Engine_& rng) {
        /*
         * We're assuming that lower < upper at each point within this
         * recursion. This requires some protection at the call site
         * when nobs = 0, see the reset() function.
         */

        Index_ pos = nodes.size();
        nodes.emplace_back();
        Node& node = nodes.back(); // this is safe during recursion because we reserved 'nodes' already to the number of observations, see reset().

        Index_ gap = upper - lower;
        if (gap > 1) { // not yet at a leaf.

            /* Choose an arbitrary point and move it to the start of the [lower, upper)
             * interval in 'items'; this is our new vantage point.
             *
             * Yes, I know that the modulo method does not provide strictly
             * uniform values but statistical correctness doesn't really matter
             * here, and I don't want std::uniform_int_distribution's
             * implementation-specific behavior.
             */
            Index_ i = (rng() % gap + lower);
            std::swap(items[lower], items[i]);
            const auto& vantage = items[lower];
            node.index = vantage.second;
            const Float_* vantage_ptr = coords + static_cast<size_t>(vantage.second) * long_num_dim; // cast to avoid overflow.
            node.center = vantage_ptr;

            // Compute distances to the new vantage point.
            for (Index_ i = lower + 1; i < upper; ++i) {
                const Float_* loc = coords + static_cast<size_t>(items[i].second) * long_num_dim; // cast to avoid overflow.
                items[i].first = raw_distance(vantage_ptr, loc, num_dim);
            }

            // Partition around the median distance from the vantage point.
            Index_ median = lower + gap/2;
            Index_ lower_p1 = lower + 1; // excluding the vantage point itself, obviously.
            std::nth_element(items.begin() + lower_p1, items.begin() + median, items.begin() + upper);

            // Radius of the new node will be the distance to the median.
            node.radius = std::sqrt(items[median].first);

            // Recursively build tree.
            if (lower_p1 < median) {
                node.left = build(lower_p1, median, coords, rng);
            }
            if (median < upper) {
                node.right = build(median, upper, coords, rng);
            }

        } else {
            const auto& leaf = items[lower];
            node.index = leaf.second;
            node.center = coords + static_cast<size_t>(leaf.second) * long_num_dim; // cast to avoid overflow.
        }

        return pos;
    }

public:
    QuickSearch() = default;

    QuickSearch(Dim_ ndim, Index_ nobs, const Float_* vals) {
        reset(ndim, nobs, vals);
    }

    void reset(Dim_ ndim, Index_ nobs, const Float_* vals) {
        num_dim = ndim;
        long_num_dim = ndim;
        items.clear();
        nodes.clear();

        if (nobs) {
            items.reserve(nobs);
            for (Index_ i = 0; i < nobs; ++i) {
                items.emplace_back(0, i);
            }

            nodes.reserve(nobs);

            // Statistical correctness doesn't matter (aside from tie breaking)
            // so we'll just use a deterministically 'random' number to ensure
            // we get the same ties for any given dataset but a different stream
            // of numbers between datasets. Casting to get well-defined overflow.
            uint64_t base = 1234567890, m1 = nobs, m2 = ndim;
            std::mt19937_64 rand(base * m1 +  m2);

            build(0, nobs, vals, rand);
        }
    }


private:
    template<typename Query_>
    void search_nn(Index_ curnode_index, const Query_* target, Index_& closest_point, Float_& closest_dist) const {
        const auto& curnode=nodes[curnode_index];
        Float_ dist = std::sqrt(raw_distance(curnode.center, target, num_dim));
        if (dist < closest_dist) {
            closest_point = curnode.index;
            closest_dist = dist;
        }

        if (dist < curnode.radius) { // If the target lies within the radius of ball:
            if (curnode.left != LEAF && dist - closest_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, closest_point, closest_dist);
            }

            if (curnode.right != LEAF && dist + closest_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, closest_point, closest_dist);
            }

        } else { // If the target lies outside the radius of the ball:
            if (curnode.right != LEAF && dist + closest_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, closest_point, closest_dist);
            }

            if (curnode.left != LEAF && dist - closest_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, closest_point, closest_dist);
            }
        }
    }

public:
    template<typename Query_>
    Index_ find(const Query_* query) const {
        Float_ closest_dist = std::numeric_limits<Float_>::max();
        Index_ closest = 0;
        search_nn(0, query, closest, closest_dist);
        return closest;
    }

    template<typename Query_>
    std::pair<Index_, Float_> find_with_distance(const Query_* query) const {
        Float_ closest_dist = std::numeric_limits<Float_>::max();
        Index_ closest = 0;
        search_nn(0, query, closest, closest_dist);
        return std::make_pair(closest, closest_dist);
    }

private:
    template<typename Query_>
    void search_nn(Index_ curnode_index, const Query_* target, std::priority_queue<std::pair<Float_, Index_> >& closest) const {
        const auto& curnode=nodes[curnode_index];
        Float_ dist = std::sqrt(raw_distance(curnode.center, target, num_dim));

        auto biggest_dist = closest.top().first;
        if (dist < biggest_dist) {
            closest.pop();
            closest.emplace(dist, curnode.index);
            biggest_dist = closest.top().first;
        }

        if (dist < curnode.radius) { // If the target lies within the radius of ball:
            if (curnode.left != LEAF && dist - biggest_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, closest);
            }

            if (curnode.right != LEAF && dist + biggest_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, closest);
            }

        } else { // If the target lies outside the radius of the ball:
            if (curnode.right != LEAF && dist + biggest_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, closest);
            }

            if (curnode.left != LEAF && dist - biggest_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, closest);
            }
        }
    }

public:
    template<typename Query_>
    std::pair<Index_, Index_> find2(const Query_* query) const {
        // There better be two or more observations in this dataset,
        // otherwise one of the placeholders will end up being reported!
        std::priority_queue<std::pair<Float_, Index_> > closest;
        closest.emplace(std::numeric_limits<Float_>::max(), 0);
        closest.emplace(std::numeric_limits<Float_>::max(), 0);
        search_nn(0, query, closest);

        std::pair<Index_, Index_> output;
        output.second = closest.top().second;
        closest.pop();
        output.first = closest.top().second;
        return output;
    }
};

}

}

#endif
