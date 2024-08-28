#ifndef KMEANS_PARALLELIZE_HPP 
#define KMEANS_PARALLELIZE_HPP

/**
 * @file parallelize.hpp
 * @brief Utilities for parallelization.
 */

#ifndef KMEANS_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"

/**
 * Function-like macro implementing the parallelization scheme for the **kmeans** library.
 * If undefined by the user, it defaults to `subpar::parallelize()`.
 * Any user-defined macro should accept the same arguments as `subpar::parallelize()`.
 */ 
#define KMEANS_CUSTOM_PARALLEL ::subpar::parallelize
#endif

#endif
