cmake_minimum_required(VERSION 3.24)

project(kmeans
    VERSION 1.0.0
    DESCRIPTION "A C++ port of the Hartigan-Wong k-means algorithm"
    LANGUAGES CXX)

add_library(kmeans INTERFACE)
add_library(ltla::kmeans ALIAS kmeans)

include(GNUInstallDirs)
target_include_directories(kmeans INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ltla_kmeans>)

target_compile_features(kmeans INTERFACE cxx_std_17)

# Dependencies
option(KMEANS_FETCH_EXTERN "Automatically fetch CppKmeans's external dependencies." ON)
if(KMEANS_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_aarand CONFIG REQUIRED)
    find_package(ltla_powerit CONFIG REQUIRED)
endif()

target_link_libraries(kmeans INTERFACE ltla::aarand ltla::powerit)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(KMEANS_TESTS "Build CppKmeans's test suite." ON)
else()
    option(KMEANS_TESTS "Build CppKmeans's test suite." OFF)
endif()
if(KMEANS_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ltla_kmeans)

install(TARGETS kmeans
    EXPORT kmeansTargets)

install(EXPORT kmeansTargets
    FILE ltla_kmeansTargets.cmake
    NAMESPACE ltla::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)
