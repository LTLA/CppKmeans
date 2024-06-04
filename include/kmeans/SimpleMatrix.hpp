#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

/**
 * @file SimpleMatrix.hpp
 * @brief Wrapper for a simple dense matrix.
 */

namespace kmeans {

/**
 * @brief A simple matrix of observations.
 *
 * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
 * It is compatible with the compile-time interface described in `MockMatrix`.
 *
 * @tparam Data_ Floating-point type for the data.
 * @tparam Index_ Integer type for the observation indices.
 */
template<typename Data_, typename Index_>
class SimpleMatrix {
public:
    /**
     * @param num_dim Number of dimensions.
     * @param num_obs Number of observations.
     * @param[in] data Pointer to an array of length `num_dim * num_obs`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(int num_dim, Index_ num_obs, const Data_* data) : my_num_dim(num_dim), my_num_obs(num_obs), my_data(data), my_long_num_dim(num_dim) {}

private:
    int my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    size_t my_long_num_dim;

public:
    /**
     * @cond
     */
    typedef Data_ data_type;

    typedef Index_ index_type;

    typedef decltype(my_num_dim) dimension_type;

    struct RandomAccessWorkspace{};

    struct ConsecutiveAccessWorkspace {
        ConsecutiveAccessWorkspace(index_type start) : at(start) {}
        index_type at;
    };

    struct IndexedAccessWorkspace {
        IndexedAccessWorkspace(const index_type* sequence) : sequence(sequence) {}
        const index_type* sequence;
        index_type at = 0;
    };

public:
    Index_ num_observations() const {
        return my_num_obs;
    }

    dimension_type num_dimensions() const {
        return my_num_dim;
    }

    RandomAccessWorkspace create_workspace() const {
        return RandomAccessWorkspace();
    }

    ConsecutiveAccessWorkspace create_workspace(index_type start, index_type) const {
        return ConsecutiveAccessWorkspace(start);
    }

    IndexedAccessWorkspace create_workspace(const index_type* sequence, index_type) const {
        return IndexedAccessWorkspace(sequence);
    }

    const data_type* get_observation(Index_ i, [[maybe_unused]] RandomAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(i) * my_long_num_dim; // avoid overflow during multiplication.
    } 

    const data_type* get_observation(ConsecutiveAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(workspace.at++) * my_long_num_dim; // avoid overflow during multiplication.
    } 

    const data_type* get_observation(IndexedAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(workspace.sequence[workspace.at++]) * my_long_num_dim; // avoid overflow during multiplication.
    } 
    /**
     * @endcond
     */
};

}

#endif
