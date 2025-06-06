#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

#include "Matrix.hpp"

/**
 * @file SimpleMatrix.hpp
 * @brief Wrapper for a simple dense matrix.
 */

namespace kmeans {

/**
 * @cond
 */
template<typename Index_, typename Data_>
class SimpleMatrix;

template<typename Index_, typename Data_>
class SimpleMatrixRandomAccessExtractor final : public RandomAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixRandomAccessExtractor(const SimpleMatrix<Index_, Data_>& parent) : my_parent(parent) {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;

public:
    const Data_* get_observation(Index_ i) {
        return my_parent.my_data + static_cast<size_t>(i) * my_parent.my_num_dim; // cast to size_t to avoid overflow during multiplication.
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixConsecutiveAccessExtractor final : public ConsecutiveAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixConsecutiveAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, size_t start) : my_parent(parent), my_position(start) {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    size_t my_position;

public:
    const Data_* get_observation() {
        return my_parent.my_data + (my_position++) * my_parent.my_num_dim; // already size_t's, no casting required.
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixIndexedAccessExtractor final : public IndexedAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixIndexedAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, const Index_* sequence) : my_parent(parent), my_sequence(sequence) {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    const Index_* my_sequence;
    size_t my_position = 0;

public:
    const Data_* get_observation() {
        return my_parent.my_data + static_cast<size_t>(my_sequence[my_position++]) * my_parent.my_num_dim; // cast to size_t to avoid overflow during multiplication.
    }
};
/**
 * @endcond
 */

/**
 * @brief A simple matrix of observations.
 *
 * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
 *
 * @tparam Data_ Numeric type for the data.
 * @tparam Index_ Integer type for the observation indices.
 */
template<typename Index_, typename Data_>
class SimpleMatrix final : public Matrix<Index_, Data_> {
public:
    /**
     * @param num_dimensions Number of dimensions.
     * @param num_observations Number of observations.
     * @param[in] data Pointer to an array of length `num_dim * num_obs`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(size_t num_dimensions, Index_ num_observations, const Data_* data) : 
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data) {}

private:
    size_t my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    friend class SimpleMatrixRandomAccessExtractor<Index_, Data_>;
    friend class SimpleMatrixConsecutiveAccessExtractor<Index_, Data_>;
    friend class SimpleMatrixIndexedAccessExtractor<Index_, Data_>;

public:
    /**
     * @cond
     */
    Index_ num_observations() const {
        return my_num_obs;
    }

    size_t num_dimensions() const {
        return my_num_dim;
    }

public:
    std::unique_ptr<RandomAccessExtractor<Index_, Data_> > new_extractor() const {
        return std::make_unique<SimpleMatrixRandomAccessExtractor<Index_, Data_> >(*this);
    }

    std::unique_ptr<ConsecutiveAccessExtractor<Index_, Data_> > new_extractor(Index_ start, Index_) const {
        return std::make_unique<SimpleMatrixConsecutiveAccessExtractor<Index_, Data_> >(*this, start);
    }

    std::unique_ptr<IndexedAccessExtractor<Index_, Data_> > new_extractor(const Index_* sequence, size_t) const {
        return std::make_unique<SimpleMatrixIndexedAccessExtractor<Index_, Data_> >(*this, sequence);
    }
    /**
     * @endcond
     */
};

}

#endif
