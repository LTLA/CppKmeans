#ifndef KMEANS_UTILS_HPP
#define KMEANS_UTILS_HPP

#include <algorithm>

namespace kmeans {

namespace internal {

template<class Chosen_, class Matrix_, typename Data_>
void copy_into_array(const Chosen_& chosen, const Matrix_& in, Data_* out) {
    auto work = in.create_workspace();
    auto ndim = in.num_dimensions();
    for (auto c : chosen) {
        auto ptr = in.fetch_observation(c, work);
        std::copy_n(ptr, ndim, out);
        out += ndim;
    }
}

template<typename Index_, class Function_>
void parallelize(Index_ ntasks, int nthreads, Function_ fun) {
    if (nthreads > 1) {
#if defined(KMEANS_CUSTOM_PARALLEL)
        KMEANS_CUSTOM_PARALLEL(ntasks, nthreads, std::move(fun));
        return;

#elif defined(_OPENMP) 
        Index_ per_thread = (ntasks / nthreads) + (ntasks % nthreads > 0);
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
            Index_ start = per_thread * t;
            Index_ length = std::min(ntasks - start, per_thread);
            fun(t, start, length);
        }
        return;
#endif
    }

    fun(0, 0, ntasks);
}

}

}

#endif
