cmake_minimum_required(VERSION 3.4...3.18)
project(_docreset)

add_subdirectory(pybind11)
pybind11_add_module(_docreset src/docreset.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_docreset PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
