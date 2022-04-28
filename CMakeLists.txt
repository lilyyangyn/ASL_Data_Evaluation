cmake_minimum_required(VERSION 3.1)

project(asl)

add_executable(straight straight.cpp)

target_compile_options(straight PRIVATE -O3 -march=native)