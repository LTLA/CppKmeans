include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing for tests")

include(GoogleTest)

macro(create_test target)
    add_executable(
        ${target} 
        src/HartiganWong.cpp
        src/Lloyd.cpp
        src/MiniBatch.cpp
        src/InitializeRandom.cpp
        src/InitializeKmeansPP.cpp
        src/InitializePCAPartition.cpp
        src/Kmeans.cpp
    )

    target_link_libraries(
        ${target}
        gtest_main
        kmeans
    )

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)

    if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

create_test(libtest)
create_test(cuspartest)
target_compile_definitions(cuspartest PRIVATE CUSTOM_PARALLEL_TEST=1)

find_package(OpenMP)
if(OpenMP_FOUND)
    create_test(omptest)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)
endif()
