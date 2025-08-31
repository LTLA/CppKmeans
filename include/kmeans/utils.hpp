#ifndef KMEANS_UTILS_HPP
#define KMEANS_UTILS_HPP

#include <type_traits>

namespace kmeans {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}

}

#endif
