#ifndef KMEANS_COPY_HPP
#define KMEANS_COPY_HPP

#include <algorithm>
#include "Matrix.hpp"

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void copy_into_array(const Matrix_& matrix, const std::vector<Index<Matrix_> >& chosen, Float_* out) {
    size_t ndim = matrix.num_dimensions();
    size_t nchosen = chosen.size();
    auto work = matrix.new_extractor(chosen.data(), nchosen);
    for (size_t i = 0; i < nchosen; ++i) {
        auto ptr = work->get_observation();
        std::copy_n(ptr, ndim, out);
        out += ndim;
    }
}

}

}

#endif
