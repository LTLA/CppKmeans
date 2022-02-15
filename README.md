# C++ library for k-means

![Unit tests](https://github.com/LTLA/CppKmeans/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/CppKmeans/actions/workflows/doxygenate.yaml/badge.svg)
![stats comparison](https://github.com/LTLA/CppKmeans/actions/workflows/compare-kmeans.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/CppKmeans/branch/master/graph/badge.svg?token=7S231XHC0Q)](https://codecov.io/gh/LTLA/CppKmeans)

## Overview

This repository contains a header-only C++ library for k-means clustering.
Initialization can be performed with user-supplied centers, random selection of points, weighted sampling with kmeans++ (Arthur and Vassilvitskii, 2007) or PCA partitioning (Su and Dy, 2007)/
Refinement can be performed using the Hartigan-Wong approach or Lloyd's algorithm.
The Hartigan-Wong implementation is derived from the Fortran code in the R **stats** package, heavily refactored for more idiomatic C++.

## Quick start

**kmeans** is a header-only library, so it can be easily used by just `#include`ing the relevant source files:

```cpp
#include "kmeans/Kmeans.hpp"

// assorted boilerplate here...

auto res = kmeans::Kmeans().run(ndim, nobs, matrix.data(), ncenters);
res.centers; // Matrix of centroid coordinates, stored in column-major format
res.clusters; // Vector of cluster assignments
res.details; // Details from the clustering algorithm
```

If you already allocated arrays for the centroids and clusters, you can fill the arrays directly:

```cpp
std::vector<double> centers(ndim * ncenters);
std::vector<int> clusters(nobs);
auto deets = kmeans::Kmeans().run(ndim, nobs, matrix.data(), ncenters, 
                                  centers.data(), clusters.data());
deets.withinss;
deets.sizes;
```

If you want to fiddle with parameters, use the relevant setters:

```cpp
kmeans::Kmeans km;
km.set_seed(42);
auto res2 = km.run(ndim, nobs, matrix.data(), ncenters);

// Or change the underlying algorithms for initialization/refinement:
kmeans::InitializeRandom rd;
kmeans::Lloyd ll;
ll.set_max_iterations(100);
auto res3 = kmeans::Kmeans().run(ndim, nobs, matrix.data(), ncenters, &rd, &ll);
```

See the [reference documentation](https://ltla.github.io/CppKmeans) for more details.

## Building projects 

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(kmeans)
```

Then you can link to **kmeans** to make the headers available during compilation:

```
# For executables:
target_link_libraries(myexe kmeans)

# For libaries
target_link_libraries(mylib INTERFACE kmeans)
```

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the additional [**aarand**](https://github.com/LTLA/aarand) library for some lightweight distribution functions.

## References

Hartigan, J. A. and Wong, M. A. (1979).
Algorithm AS 136: A K-means clustering algorithm.
_Applied Statistics_ 28, 100-108.

Arthur, D. and Vassilvitskii, S. (2007). 
k-means++: the advantages of careful seeding.
_Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.

Su, T. and Dy, J. G. (2007).
In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
_Intelligent Data Analysis_ 11, 319-338.
