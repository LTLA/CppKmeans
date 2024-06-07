# C++ library for k-means

![Unit tests](https://github.com/LTLA/CppKmeans/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/CppKmeans/actions/workflows/doxygenate.yaml/badge.svg)
![stats comparison](https://github.com/LTLA/CppKmeans/actions/workflows/compare-kmeans.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/CppKmeans/branch/master/graph/badge.svg?token=7S231XHC0Q)](https://codecov.io/gh/LTLA/CppKmeans)

## Overview

This repository contains a header-only C++ library for k-means clustering.
Initialization can be performed with user-supplied centers, random selection of points, weighted sampling with kmeans++ (Arthur and Vassilvitskii, 2007) or PCA partitioning (Su and Dy, 2007).
Refinement can be performed using the Hartigan-Wong approach or Lloyd's algorithm.
The Hartigan-Wong implementation is derived from the Fortran code in the R **stats** package, heavily refactored for more idiomatic C++.

## Quick start

**kmeans** is a header-only library, so it can be easily used by just `#include`ing the relevant source files:

```cpp
#include "kmeans/Kmeans.hpp"

// assorted boilerplate here...

// Wrap your matrix in a SimpleMatrix.
kmeans::SimpleMatrix kmat(ndim, nobs, matrix.data());

auto res = kmeans::compute(
    kmat,
    kmeans::InitializeKmeanspp(), // initialize with kmeans++
    kmeans::RefineLloyd(), // refine with Lloyd's algorithm
    ncenters 
);

res.centers; // Matrix of centroid coordinates, stored in column-major format
res.clusters; // Vector of cluster assignments
res.details; // Details from the clustering algorithm

// Compute the WCSS if we want it:
std::vector<double> wcss(ncenters);
kmeans::compute_wcss(
    kmat, 
    ncenters, 
    res.centers.data(), 
    res.clusters.data(), 
    wcss.data()
);
```

If we already allocated arrays for the centroids and clusters, we can fill the arrays directly:

```cpp
std::vector<double> centers(ndim * ncenters);
std::vector<int> clusters(nobs);

auto deets = kmeans::compute(
    kmat,
    kmeans::InitializeRandom(), // random initialization
    kmeans::RefineHartiganWong(), // refine with Hartigan-Wong 
    ncenters 
    centers.data(),
    clusters.data()
);
```

We can tune the clustering by passing options into the constructors of the relevant classes:

```cpp
kmeans::InitializePcaPartitionOptions pp_opt;
pp_opt.power_iteration_options.iterations = 200;
pp_opt.seed = 42;
kmeans::InitializePcaPartitions pp(pp_opt);

kmeans::RefineLloydOptions ll_opt;
ll_opt.max_iterations = 10;
ll_opt.num_threads = 3;
kmeans::RefineLloyd ll(ll_opt);

auto res2 = kmeans::compute(kmat, pp, ll, ncenters);
```

See the [reference documentation](https://ltla.github.io/CppKmeans) for more details.

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(kmeans)
```

Then you can link to **kmeans** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe ltla::kmeans)

# For libaries
target_link_libraries(mylib INTERFACE ltla::kmeans)
```

### CMake with `find_package()`

To install the library, clone an appropriate version of this repository and run:

```sh
mkdir build && cd build
cmake .. -DKMEANS_TESTS=OFF
cmake --build . --target install
```

Then we can use `find_package()` as usual:

```cmake
find_package(ltla_kmeans CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE ltla::kmeans)
```

By default, this will use `FetchContent` to fetch all external dependencies (see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for a list).
If you want to install them manually, use `-DKMEANS_FETCH_EXTERN=OFF`.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt), which also need to be made available during compilation.

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
