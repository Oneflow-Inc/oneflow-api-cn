cmake_minimum_required(VERSION 3.14.0)
project(_docreset)

execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 REQUIRED)
pybind11_add_module(_docreset src/docreset.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_docreset PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
