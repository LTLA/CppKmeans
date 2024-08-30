#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#ifdef TEST_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#define KMEANS_CUSTOM_PARALLEL(nw, nt, fun) subpar::parallelize_range(nw, nt, fun)
#endif
