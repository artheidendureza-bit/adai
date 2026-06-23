cmake_minimum_required(VERSION 3.15)
project(adaicpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(ADAICPP_BUILD_TESTS "Build tests" ON)
option(ADAICPP_ENABLE_CUDA "Enable CUDA support" OFF)
option(ADAICPP_BUILD_PYTHON "Build Python bindings" OFF)

if(ADAICPP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(CORE_SOURCES
    src/core/device.cpp
    src/core/tensor.cpp
    src/core/memory_pool.cpp
)

set(GRAPH_SOURCES
    src/graph/node.cpp
    src/graph/graph.cpp
)

set(OPS_SOURCES
    src/ops/op_registry.cpp
)

set(ALL_SOURCES ${CORE_SOURCES} ${GRAPH_SOURCES} ${OPS_SOURCES})

if(ADAICPP_ENABLE_CUDA)
    set(CUDA_SOURCES
        src/cuda/kernels.cu
    )
    set(ALL_SOURCES ${ALL_SOURCES} ${CUDA_SOURCES})
endif()

# Create library
add_library(adaicpp STATIC ${ALL_SOURCES})

if(ADAICPP_ENABLE_CUDA)
    target_include_directories(adaicpp PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(adaicpp PRIVATE ${CUDA_LIBRARIES})
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(adaicpp PRIVATE /W4 /wd4100 /wd4267)
else()
    target_compile_options(adaicpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
if(ADAICPP_BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    
    if(GTest_FOUND)
        add_executable(adaicpp_tests tests/test_tensor.cpp tests/test_graph.cpp)
        target_link_libraries(adaicpp_tests PRIVATE adaicpp GTest::GTest)
        
        include(GoogleTest)
        gtest_discover_tests(adaicpp_tests)
    else()
        message(WARNING "GTest not found, tests will not be built")
    endif()
endif()

# Example
add_executable(basic_example examples/basic_example.cpp)
target_link_libraries(basic_example PRIVATE adaicpp)

add_executable(simple_test examples/simple_test.cpp)
target_link_libraries(simple_test PRIVATE adaicpp)

# Python bindings
if(ADAICPP_BUILD_PYTHON)
    # Find Python and pybind11
    find_package(Python3 COMPONENTS Interpreter Development)
    find_package(pybind11)
    
    if(Python3_FOUND AND pybind11_FOUND)
        # Create Python module
        pybind11_add_module(adaicpp_py
            python_bindings/tensor_bindings.cpp
            python_bindings/device_bindings.cpp
            python_bindings/node_bindings.cpp
            python_bindings/graph_bindings.cpp
            python_bindings/op_registry_bindings.cpp
            python_bindings/module.cpp
        )
        target_link_libraries(adaicpp_py PRIVATE adaicpp)
        
        if(ADAICPP_ENABLE_CUDA)
            target_include_directories(adaicpp_py PRIVATE ${CUDA_INCLUDE_DIRS})
            target_link_libraries(adaicpp_py PRIVATE ${CUDA_LIBRARIES})
        endif()
        
        # Set output directory to match Python package structure
        set_target_properties(adaicpp_py PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/adAI
            PREFIX ""
            OUTPUT_NAME "adaicpp_py"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/adAI
            SUFFIX ".pyd"
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/adAI
        )
        
        message(STATUS "Python bindings will be built")
    else()
        message(WARNING "Python3 or pybind11 not found, Python bindings will not be built")
    endif()
endif()

# Installation
install(TARGETS adaicpp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
