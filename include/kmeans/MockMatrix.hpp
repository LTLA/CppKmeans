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
 * Long story short, the matrix should support extraction of the vector of coordinates for each observation.
 */
class MockMatrix {
public:
    /**
     * @cond
     */
    MockMatrix(int num_dim, Index_ num_obs, const Data_* data) : my_num_dim(num_dim), my_num_obs(num_obs), my_data(data), my_long_num_dim(num_dim) {}
    /**
     * @endcond
     */

private:
    int my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    size_t my_long_num_dim;

public:
    /**
     * Type of the data.
     * Any floating-point type may be used here.
     */
    typedef double data_type;

    /**
     * Type for the observation indices.
     * Any integer type may be used here.
     */
    typedef size_t index_type;

    /**
     * Integer type for the dimension indices.
     * Any integer type may be used here.
     */
    typedef int dimension_type;

public:
    /**
     * @brief Workspace for random access to observations.
     *
     * This may be used by matrix implementations to store temporary data structures that can be re-used in each call to `fetch_observation()`.
     */
    struct RandomAccessWorkspace {};

    /**
     * @return A new random-access workspace, to be passed to `fetch_observation()`.
     */
    RandomAccessWorkspace create_workspace() const {
        return RandomAccessWorkspace();
    }

    /**
     * @brief Workspace for access to consecutive observations.
     *
     * This may be used by matrix implementations to store temporary data structures that can be re-used in each call to `fetch_observation()`.
     */
    struct ConsecutiveAccessWorkspace {
        /**
         * @cond
         */
        ConsecutiveAccessWorkspace(index_type start, index_type length) : at(start), end(start + length) {}
        index_type at, end;
        /**
         * @endcond
         */
    };

    /**
     * @param start Start of the contiguous block to be accessed consecutively.
     * @param length Length of the contiguous block to be accessed consecutively.
     * @return A new consecutive-access workspace, to be passed to `fetch_observation()`.
     */
    ConsecutiveWorkspace create_workspace(index_type start, index_type length) const {
        return ConsecutiveAccessWorkspace(start, length);
    }


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

    /**
     * @param i Index of the observation to fetch.
     * @param workspace Random-access workspace for fetching.
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for this observation.
     */
    const data_type* get_observation(Index_ i, [[maybe_unused]] RandomAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(i) * my_long_num_dim; // avoid overflow during multiplication.
    } 

    /**
     * @param workspace Consecutive-access workspace. 
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for this observation.
     */
    const data_type* get_observation(ConsecutiveAccessWorkspace& workspace) const {
        return my_data + static_cast<size_t>(workspace.at++) * my_long_num_dim; // avoid overflow during multiplication.
    } 
};

}

#endif
