#ifndef KMEANS_QUICKSEARCH_HPP
#define KMEANS_QUICKSEARCH_HPP

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <tuple>
#include <iostream>

namespace kmeans {

/* Adapted from http://stevehanov.ca/blog/index.php?id=130 */
template<typename DATA_t = double, typename CLUSTER_t = int> 
class QuickSearch {
private:
    int num_dim;
    CLUSTER_t num_obs;
    const DATA_t* reference;

    static DATA_t normalize(DATA_t x) {
        return std::sqrt(x);
    }

    static DATA_t raw_distance(const DATA_t* x, const DATA_t* y, int ndim) {
        DATA_t output = 0;
        for (int i = 0; i < ndim; ++i, ++x, ++y) {
            output += (*x - *y) * (*x - *y);
        }
        return output;
    }

private:
    typedef int NodeIndex_t;
    static const NodeIndex_t LEAF_MARKER=-1;

    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    struct Node {
        DATA_t threshold;  // radius 
        CLUSTER_t index; // original index of current vantage point 
        NodeIndex_t left;  // node index of the next vantage point for all children closer than 'threshold' from the current vantage point
        NodeIndex_t right; // node index of the next vantage point for all children further than 'threshold' from the current vantage point
        Node(NodeIndex_t i=0) : threshold(0), index(i), left(LEAF_MARKER), right(LEAF_MARKER) {}
    };
    std::vector<Node> nodes;

    typedef std::tuple<CLUSTER_t, const DATA_t*, DATA_t> DataPoint; // internal distances computed using "DATA_t" type, even if output is returned with DISTANCE_t.

    template<class SAMPLER>
    NodeIndex_t buildFromPoints(NodeIndex_t lower, NodeIndex_t upper, std::vector<DataPoint>& items, SAMPLER& rng) {
        if (upper == lower) {     // indicates that we're done here!
            return LEAF_MARKER;
        }

        NodeIndex_t pos = nodes.size();
        nodes.resize(pos + 1);
        Node& node=nodes.back();
            
        NodeIndex_t gap = upper - lower;
        if (gap > 1) {      // if we did not arrive at leaf yet

            /* Choose an arbitrary point and move it to the start of the [lower, upper)
             * interval in 'items'; this is our new vantage point.
             * 
             * Yes, I know that the modulo method does not provide strictly
             * uniform values but statistical correctness doesn't really matter
             * here... but reproducibility across platforms does matter, and
             * std::uniform_int_distribution is implementation-dependent!
             */
            NodeIndex_t i = static_cast<NodeIndex_t>(rng() % gap + lower);
            std::swap(items[lower], items[i]);
            const auto& vantage = items[lower];

            // Compute distances to the new vantage point.
            const DATA_t* ref = std::get<1>(vantage);
            for (NodeIndex_t i = lower + 1; i < upper; ++i) {
                const DATA_t* loc = std::get<1>(items[i]);
                std::get<2>(items[i]) = raw_distance(ref, loc, num_dim);
            }

            // Partition around the median distance from the vantage point.
            NodeIndex_t median = lower + gap/2;
            std::nth_element(items.begin() + lower + 1, items.begin() + median, items.begin() + upper,
                [&](const DataPoint& left, const DataPoint& right) -> bool {
                    return std::get<2>(left) < std::get<2>(right);
                }
            );
           
            // Threshold of the new node will be the distance to the median
            node.threshold = normalize(std::get<2>(items[median]));

            // Recursively build tree
            node.index = std::get<0>(vantage);
            node.left = buildFromPoints(lower + 1, median, items, rng);
            node.right = buildFromPoints(median, upper, items, rng);
        } else {
            node.index = std::get<0>(items[lower]);
        }
        
        return pos;
    }

private:
    template<typename INPUT_t>
    void search_nn(NodeIndex_t curnode_index, const INPUT_t* target, CLUSTER_t& closest, DATA_t& tau) const { 
        if (curnode_index == LEAF_MARKER) { // indicates that we're done here
            return;
        }
        
        // Compute distance between target and current node
        const auto& curnode=nodes[curnode_index];
        DATA_t dist = normalize(raw_distance(reference + curnode.index * num_dim, target, num_dim));

        // If current node within radius tau
        if (dist < tau) {
            closest = curnode.index;
            tau = dist;
        }

        // Return if we arrived at a leaf
        if (curnode.left == LEAF_MARKER && curnode.right == LEAF_MARKER) {
            return;
        }
        
        // If the target lies within the radius of ball
        if (dist < curnode.threshold) {
            if (dist - tau <= curnode.threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, closest, tau);
            }
            
            if (dist + tau >= curnode.threshold) {         // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, closest, tau);
            }
        
        // If the target lies outsize the radius of the ball
        } else {
            if (dist + tau >= curnode.threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, closest, tau);
            }
            
            if (dist - tau <= curnode.threshold) {         // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, closest, tau);
            }
        }
    }

public:
    QuickSearch(CLUSTER_t ndim, CLUSTER_t nobs, const DATA_t* vals) : num_dim(ndim), num_obs(nobs), reference(vals) {
        std::vector<DataPoint> items;
        items.reserve(num_obs);
        auto ptr = vals;
        for (CLUSTER_t i = 0; i < num_obs; ++i, ptr += num_dim) {
            items.push_back(DataPoint(i, ptr, 0));
        }

        nodes.reserve(num_obs);
        std::mt19937_64 rand(1234567890); // seed doesn't really matter, we don't need statistical correctness here.
        buildFromPoints(0, num_obs, items, rand);
        return;
    }

    CLUSTER_t find(const DATA_t* query) const {
        DATA_t tau = std::numeric_limits<DATA_t>::max();
        CLUSTER_t closest = 0;
        search_nn(0, query, closest, tau);
        return closest;
    }

    std::pair<CLUSTER_t, DATA_t> find_with_distance(const DATA_t* query) const {
        DATA_t tau = std::numeric_limits<DATA_t>::max();
        CLUSTER_t closest = 0;
        search_nn(0, query, closest, tau);
        return std::make_pair(closest, tau);
    }
};

}

#endif
