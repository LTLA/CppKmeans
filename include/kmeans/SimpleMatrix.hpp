#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

namespace kmeans {

template<typename Data_, typename Index_>
class SimpleMatrix {
public:
    SimpleMatrix(int num_dim, Index_ num_obs, const Data_* data) : my_num_dim(num_dim), my_num_obs(num_obs), my_data(data), my_long_num_dim(num_dim) {}

private:
    int my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    size_t my_long_num_dim;

public:
    typename Data_ data_type;

    typedef Index_ index_type;

    struct Workspace {};

public:
    Index_ num_obs() const {
        return my_num_obs;
    }

    int num_dim() const {
        return my_num_dim;
    }

public:
    Workspace create_workspace() const {
        return Workspace();
    }

    const Data_* get_observation(Index_ i, [[maybe_unused]] Workspace& workspace) const {
        return my_data + static_cast<size_t>(i) * my_long_num_dim; // avoid overflow during multiplication.
    } 
}

}

#endif
