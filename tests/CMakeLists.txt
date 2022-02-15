include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest 
    src/HartiganWong.cpp
    src/Lloyd.cpp
    src/MiniBatch.cpp
    src/InitializeRandom.cpp
    src/InitializeKmeansPP.cpp
    src/InitializePCAPartition.cpp
    src/Kmeans.cpp
)

target_link_libraries(
    libtest
    gtest_main
    kmeans
)

set(USE_OPENMP OFF CACHE BOOL "Enable OpenMP for tests")
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(libtest OpenMP::OpenMP_CXX)
endif()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing for tests")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)

add_test(NAME kmeans_tests COMMAND libtest)
