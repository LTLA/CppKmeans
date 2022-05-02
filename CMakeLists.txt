cmake_minimum_required(VERSION 3.14)

project(kmeans
    VERSION 1.0.0
    DESCRIPTION "A C++ port of the Hartigan-Wong k-means algorithm"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(kmeans INTERFACE)

add_subdirectory(extern)

target_link_libraries(kmeans INTERFACE aarand powerit)

target_include_directories(kmeans INTERFACE include/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()
