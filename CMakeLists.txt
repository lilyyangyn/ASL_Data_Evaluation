cmake_minimum_required(VERSION 3.1)

project(asl)

set(CMAKE_CXX_STANDARD 17)

if (COUNT_FLOPS)
    add_compile_options(-DFLOPS)
endif()

add_library(benchmark STATIC benchmark.cpp)
add_library(exact STATIC exact.cpp)
add_library(improved_mc STATIC improved_mc.cpp)
add_library(flops STATIC flops.cpp)

add_executable(measure measure.cpp)


enable_testing()

add_executable(test_exact tests/test_exact.cpp)
add_test(test_exact test_exact)

add_executable(test_improved_mc tests/test_improved_mc.cpp)
add_test(test_improved_mc test_improved_mc)

target_include_directories(exact PUBLIC include)
target_include_directories(improved_mc PUBLIC include)
target_include_directories(flops PUBLIC include)
target_include_directories(benchmark PUBLIC include)

# target_compile_options(measure PRIVATE -march=native)
target_link_libraries(exact PUBLIC flops)
target_link_libraries(improved_mc PUBLIC flops)
target_link_libraries(measure PRIVATE benchmark exact improved_mc)
target_link_libraries(test_exact PRIVATE exact)
target_compile_definitions(test_exact PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(test_improved_mc PRIVATE improved_mc)
target_compile_definitions(test_improved_mc PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests")
