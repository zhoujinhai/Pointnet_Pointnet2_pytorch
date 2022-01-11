cmake_minimum_required(VERSION 3.9)

project(seg VERSION 1.0)

# >>> build type 
# set(CMAKE_BUILD_TYPE "Debug")				# 指定生成的版本
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< CXX11

set(OPENCV_ROOT "D:/opencv-4.5.4/Release")
# set(OPENCV_ROOT "D:/opencv-4.5.4/Debug")
include_directories(${OPENCV_ROOT}/include)

if(WIN32)
	link_directories(${OPENCV_ROOT}/lib/x64/)
	# message(${OPENCV_ROOT}/lib/x64/)
	add_executable(${PROJECT_NAME} test.cpp)
	target_link_libraries(${PROJECT_NAME} opencv_world454)   # release
	# target_link_libraries(${PROJECT_NAME} opencv_world454d)  # debug 

endif()

    


