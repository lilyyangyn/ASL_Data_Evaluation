cmake_minimum_required(VERSION 3.1)

project(asl)

add_executable(exact exact.cpp)

target_compile_options(exact PRIVATE -march=native)