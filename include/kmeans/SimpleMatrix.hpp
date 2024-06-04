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
    typename Data_ data_type;

    typedef Index_ index_type;

    typedef size_t dimension_type;

    struct Workspace{};

    Index_ num_obs() const {
        return my_num_obs;
    }

    int num_dim() const {
        return my_num_dim;
    }

    Workspace create_workspace() const {
        return Workspace();
    }

    const Data_* get_observation(Index_ i, [[maybe_unused]] Workspace& workspace) const {
        return my_data + static_cast<size_t>(i) * my_long_num_dim; // avoid overflow during multiplication.
    } 
    /**
     * @endcond
     */
};

}

#endif
