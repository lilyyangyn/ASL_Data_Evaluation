cmake_minimum_required(VERSION 3.1)

project(asl)

add_library(benchmark STATIC benchmark.cpp)

add_executable(exact exact.cpp)

target_include_directories(exact PRIVATE include)
target_compile_options(exact PRIVATE -march=native)
target_link_libraries(exact benchmark)