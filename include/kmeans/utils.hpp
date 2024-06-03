#ifndef KMEANS_UTILS_HPP
#define KMEANS_UTILS_HPP

namespace kmeans {

namespace internal {

template<class Chosen_, typename Data_>
void copy_into_array(const Chosen_& chosen, size_t ndim, const Data_* in, Data_* out) {
    for (auto c : chosen) {
        auto ptr = in + static_cast<size_t>(c) * ndim; // cast to avoid overflow.
        std::copy(ptr, ptr + ndim, out);
        out += ndim;
    }
}

}

}

#endif
