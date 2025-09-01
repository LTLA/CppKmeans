#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

#include <cstddef>

#include "sanisizer/sanisizer.hpp"

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
    const Data_* get_observation(const Index_ i) {
        return my_parent.my_data + sanisizer::product_unsafe<std::size_t>(i, my_parent.my_num_dim);
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixConsecutiveAccessExtractor final : public ConsecutiveAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixConsecutiveAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, const Index_ start) : my_parent(parent), my_position(start) {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    Index_ my_position;

public:
    const Data_* get_observation() {
        return my_parent.my_data + sanisizer::product_unsafe<std::size_t>(my_position++, my_parent.my_num_dim);
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixIndexedAccessExtractor final : public IndexedAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixIndexedAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, const Index_* const sequence) : my_parent(parent), my_sequence(sequence) {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    const Index_* my_sequence;
    std::size_t my_position = 0;

public:
    const Data_* get_observation() {
        return my_parent.my_data + sanisizer::product_unsafe<std::size_t>(my_sequence[my_position++], my_parent.my_num_dim);
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
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Data_ Numeric type of the data.
 */
template<typename Index_, typename Data_>
class SimpleMatrix final : public Matrix<Index_, Data_> {
public:
    /**
     * @param num_dimensions Number of dimensions.
     * @param num_observations Number of observations.
     * @param[in] data Pointer to an array of length equal to the product of `num_dimensions` and `num_observations`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(const std::size_t num_dimensions, const Index_ num_observations, const Data_* const data) :
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data) {}

private:
    std::size_t my_num_dim;
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

    std::size_t num_dimensions() const {
        return my_num_dim;
    }

public:
    std::unique_ptr<RandomAccessExtractor<Index_, Data_> > new_extractor() const {
        return std::make_unique<SimpleMatrixRandomAccessExtractor<Index_, Data_> >(*this);
    }

    std::unique_ptr<ConsecutiveAccessExtractor<Index_, Data_> > new_extractor(const Index_ start, const Index_) const {
        return std::make_unique<SimpleMatrixConsecutiveAccessExtractor<Index_, Data_> >(*this, start);
    }

    std::unique_ptr<IndexedAccessExtractor<Index_, Data_> > new_extractor(const Index_* sequence, const std::size_t) const {
        return std::make_unique<SimpleMatrixIndexedAccessExtractor<Index_, Data_> >(*this, sequence);
    }
    /**
     * @endcond
     */
};

}

#endif
