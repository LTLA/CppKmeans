#ifndef KMEANS_QUICKSEARCH_HPP
#define KMEANS_QUICKSEARCH_HPP

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <tuple>
#include <iostream>

namespace kmeans {

namespace internal {

/* Adapted from http://stevehanov.ca/blog/index.php?id=130 */
template<typename Data_, typename Index_, typename Dim_>
class QuickSearch {
private:
    Dim_ num_dim;
    size_t long_num_dim;

    template<typename Query_>
    static Data_ raw_distance(const Data_* x, const Query_* y, Dim_ ndim) {
        Data_ output = 0;
        for (Dim_ i = 0; i < ndim; ++i, ++x, ++y) {
            Data_ delta = *x - *y;
            output += delta * delta;
        }
        return output;
    }

private:
    typedef std::pair<Data_, Index_> DataPoint; 
    std::vector<DataPoint> items;

    // Index_ might be unsigned, so we use zero as the LEAF marker.
    static const Index_ LEAF = 0;
    struct Node {
        const Data_* center = NULL;
        Data_ radius = 0; 

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

    template<class Engine_>
    Index_ build(Index_ lower, Index_ upper, const Data_* coords, Engine_& rng) {
        if (upper == lower) {     // indicates that we're done here!
            return LEAF;
        }

        Index_ pos = nodes.size();
        nodes.resize(pos + 1);
        Node& node = nodes.back();
            
        Index_ gap = upper - lower;
        if (gap > 1) {      // if we did not arrive at leaf yet

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
            const Data_* vantage_ptr = coords + static_cast<size_t>(vantage.second) * long_num_dim; // cast to avoid overflow.
            node.center = vantage_ptr;

            // Compute distances to the new vantage point.
            for (Index_ i = lower + 1; i < upper; ++i) {
                const Data_* loc = coords + static_cast<size_t>(items[i].second) * long_num_dim; // cast to avoid overflow.
                items[i].first = raw_distance(vantage_ptr, loc, num_dim);
            }

            // Partition around the median distance from the vantage point.
            Index_ median = lower + gap/2;
            std::nth_element(items.begin() + lower + 1, items.begin() + median, items.begin() + upper);

            // Radius of the new node will be the distance to the median.
            node.radius = std::sqrt(items[median].first);

            // Recursively build tree. 
            node.left = build(lower + 1, median, coords, rng);
            node.right = build(median, upper, coords, rng);

        } else {
            const auto& leaf = items[lower];
            node.index = leaf.second;
            node.center = coords + static_cast<size_t>(leaf.second) * static_cast<size_t>(num_dim); // cast to avoid overflow.
        }

        return pos;
    }

private:
    template<typename Query_>
    void search_nn(Index_ curnode_index, const Query_* target, Index_& closest, Data_& tau) const { 
        if (curnode_index == LEAF) { // indicates that we're done here
            return;
        }
        
        // Compute distance between target and current node
        const auto& curnode=nodes[curnode_index];
        Data_ dist = std::sqrt(raw_distance(curnode.center, target, num_dim));

        // If current node within radius tau
        if (dist < tau) {
            closest = curnode.index;
            tau = dist;
        }

        // Return if we arrived at a leaf
        if (curnode.left == LEAF && curnode.right == LEAF) {
            return;
        }
        
        if (dist < curnode.radius) { // If the target lies within the radius of ball:
            if (dist - tau <= curnode.radius) {         // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, closest, tau);
            }
            
            if (dist + tau >= curnode.radius) {         // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, closest, tau);
            }
        
        } else { // If the target lies outside the radius of the ball:
            if (dist + tau >= curnode.radius) {         // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, closest, tau);
            }
            
            if (dist - tau <= curnode.radius) {         // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, closest, tau);
            }
        }
    }

public:
    QuickSearch() = default;

    QuickSearch(Dim_ ndim, Index_ nobs, const Data_* vals) {
        reset(ndim, nobs, vals);
    }

    void reset(Dim_ ndim, Index_ nobs, const Data_* vals) {
        num_dim = ndim;
        long_num_dim = ndim;

        items.clear();
        items.reserve(nobs);
        for (Index_ i = 0; i < nobs; ++i) {
            items.emplace_back(0, i);
        }

        nodes.clear();
        std::mt19937_64 rand(1234567890u * nobs + ndim); // statistical correctness doesn't matter so we'll just use a deterministicly 'random' number.
        build(0, nobs, vals, rand);
    }

    template<typename Query_>
    Index_ find(const Query_* query) const {
        Data_ tau = std::numeric_limits<Data_>::max();
        Index_ closest = 0;
        search_nn(0, query, closest, tau);
        return closest;
    }

    template<typename Query_>
    std::pair<Index_, Data_> find_with_distance(const Query_* query) const {
        Data_ tau = std::numeric_limits<Data_>::max();
        Index_ closest = 0;
        search_nn(0, query, closest, tau);
        return std::make_pair(closest, tau);
    }
};

}

}

#endif
