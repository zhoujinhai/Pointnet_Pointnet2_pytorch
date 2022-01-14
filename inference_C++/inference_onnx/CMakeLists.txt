# ref: https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

# windows10

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(ops)

set(CMAKE_VERBOSE_MAKEFILE ON)


# Torch
set(TORCH_ROOT "/home/heygears/jinhai_zhou/package/libtorch")  # "E:/code/LibTorch/libtorch_debug")   
include_directories(${TORCH_ROOT}/include)
link_directories(${TORCH_ROOT}/lib/)

# Define our library target
add_library(${PROJECT_NAME} SHARED ops.cpp)
# add_executable(${PROJECT_NAME} ops.cpp)

# Enable C++14
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

# Link against Torch
target_link_libraries(${PROJECT_NAME} 
	c10
	torch
)


