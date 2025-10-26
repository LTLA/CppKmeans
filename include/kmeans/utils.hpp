#ifndef KMEANS_UTILS_HPP
#define KMEANS_UTILS_HPP

#include <type_traits>

namespace kmeans {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

}

#endif
