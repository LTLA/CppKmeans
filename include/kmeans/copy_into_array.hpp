#ifndef KMEANS_COPY_HPP
#define KMEANS_COPY_HPP

#include <algorithm>

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void copy_into_array(const Matrix_& matrix, const std::vector<typename Matrix_::index_type>& chosen, Float_* out) {
    auto work = matrix.create_workspace(chosen.data(), chosen.size());
    auto ndim = matrix.num_dimensions();
    for (size_t i = 0, end = chosen.size(); i < end; ++i) {
        auto ptr = matrix.get_observation(work);
        std::copy_n(ptr, ndim, out);
        out += ndim;
    }
}

}

}

#endif
