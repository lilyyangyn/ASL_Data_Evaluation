cmake_minimum_required(VERSION 3.1)

project(asl)

set(CMAKE_CXX_STANDARD 17)
add_library(benchmark STATIC benchmark.cpp)
add_library(exact STATIC exact.cpp)

add_executable(measure measure.cpp)

enable_testing()

add_executable(test_exact tests/test_exact.cpp)
add_test(test_exact test_exact)

target_include_directories(exact PUBLIC include)

target_compile_options(measure PRIVATE -march=native)
target_link_libraries(measure PRIVATE benchmark exact)
target_link_libraries(test_exact PRIVATE exact)