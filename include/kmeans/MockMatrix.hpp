#ifndef KMEANS_MOCK_MATRIX_HPP
#define KMEANS_MOCK_MATRIX_HPP

/**
 * @file MockMatrix.hpp
 * @brief Expectations for matrix inputs.
 */

namespace kmeans {

/**
 * @brief Compile-time interface for matrix data.
 *
 * This defines the expectations for a matrix of observation-level data to be used in `Initialize::run()` and `Refine::run()`.
 * Each matrix should support extraction of the vector of coordinates for each observation.
 */
class MockMatrix {
public:
    /**
     * @cond
     */
    MockMatrix(int num_dim, int num_obs, const double* data) : my_num_dim(num_dim), my_num_obs(num_obs), my_data(data), my_long_num_dim(num_dim) {}
    /**
     * @endcond
     */

public:
    /**
     * Type of the data.
     * Any floating-point type may be used here.
     * Note that this type is only used for data storage; all calculations are done using the `Float_` type in `Refine` and `Initialize`.
     */
    typedef double data_type;

    /**
     * Type for the observation indices.
     * Any integer type may be used here, as long as it is large enough to hold the number of observations.
     */
    typedef int index_type;

    /**
     * Integer type for the dimension indices.
     * Any integer type may be used here, as long as it is large enough to hold the number of dimensions..
     */
    typedef int dimension_type;

private:
    dimension_type my_num_dim;
    index_type my_num_obs;
    const data_type* my_data;
    size_t my_long_num_dim;

public:
    /**
     * @return Number of observations.
     */
    index_type num_observations() const {
        return my_num_obs;
    }

    /**
     * @return Number of dimensions.
     */
    dimension_type num_dimensions() const {
        return my_num_dim;
    }

public:
    /**
     * @brief Workspace for random access to observations.
     *
     * This may be used by matrix implementations to store temporary data structures that can be re-used in each call to `get_observation()`.
     */
    struct RandomAccessWorkspace {};

    /**
     * @return A new random-access workspace, to be passed to `get_observation()`.
     */
    RandomAccessWorkspace create_workspace() const {
        return RandomAccessWorkspace();
    }

    /**
     * @brief Workspace for access to consecutive observations.
     *
     * This may be used by matrix implementations to store temporary data structures that can be re-used in each call to `get_observation()`.
     */
    struct ConsecutiveAccessWorkspace {
        /**
         * @cond
         */
        ConsecutiveAccessWorkspace(index_type start) : at(start) {}
        size_t at;
        /**
         * @endcond
         */
    };

    /**
     * @param start Start of the contiguous block to be accessed consecutively.
     * @param length Length of the contiguous block to be accessed consecutively.
     * @return A new consecutive-access workspace, to be passed to `get_observation()`.
     */
    ConsecutiveAccessWorkspace create_workspace(index_type start, [[maybe_unused]] index_type length) const {
        return ConsecutiveAccessWorkspace(start);
    }

    /**
     * @brief Workspace for access to a indexed subset of observations.
     *
     * This may be used by matrix implementations to store temporary data structures that can be re-used in each call to `get_observation()`.
     */
    struct IndexedAccessWorkspace {
        /**
         * @cond
         */
        IndexedAccessWorkspace(const index_type* sequence) : sequence(sequence) {}
        const index_type* sequence;
        size_t at = 0;
        /**
         * @endcond
         */
    };

    /**
     * @param[in] sequence Pointer to an array of sorted and unique indices of observations, to be accessed in the provided order.
     * It is assumed that the vector will not be deallocated before the destruction of the returned `IndexedAccessWorkspace`.
     * @param length Number of observations in `sequence`.
     * @return A new indexed-access workspace, to be passed to `get_observation()`.
     */
    IndexedAccessWorkspace create_workspace(const index_type* sequence, [[maybe_unused]] index_type length) const {
        return IndexedAccessWorkspace(sequence);
    }

public:
    /**
     * @param i Index of the observation .
     * @param workspace Random-access workspace. 
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for observation `i`.
     */
    const data_type* get_observation(int i, [[maybe_unused]] RandomAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(i) * my_long_num_dim; // avoid overflow during multiplication.
    } 

    /**
     * @param workspace Consecutive access workspace. 
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for the next observation.
     */
    const data_type* get_observation(ConsecutiveAccessWorkspace& workspace) const {
        return my_data + (workspace.at++) * my_long_num_dim; // everything is already a size_t.
    } 

    /**
     * @param workspace Indexed access workspace. 
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for the next observation in the sequence.
     */
    const data_type* get_observation(IndexedAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(workspace.sequence[workspace.at++]) * my_long_num_dim; // avoid overflow during multiplication.
    } 
};

}

#endif
