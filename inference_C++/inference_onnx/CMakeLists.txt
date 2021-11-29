# ref: https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

# windows10

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(test_custom)

set(CMAKE_VERBOSE_MAKEFILE ON)


# Torch
set(TORCH_ROOT "D:/python/Lib/site-packages/torch")  # "D:/Anaconda3/envs/Test374/Lib/site-packages/torch")   
include_directories(${TORCH_ROOT}/include)
link_directories(${TORCH_ROOT}/lib/)

# Define our library target
add_library(${PROJECT_NAME} SHARED fps.cpp)
#add_executable(${PROJECT_NAME} fps.cpp)

# Enable C++14
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

# Link against Torch
target_link_libraries(${PROJECT_NAME} 
	c10
	torch_cpu
)


