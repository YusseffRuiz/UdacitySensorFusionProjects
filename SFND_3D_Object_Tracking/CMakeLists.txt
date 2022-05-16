# This file handles building each task as sub-project.
#
# Each task becomes an independent project in which you can experiment.
#
# Tasks are added as separate sub-projects so as compilation errors in
# one task don't affect another task.

cmake_minimum_required(VERSION 3.0.2)
project(3D_object_tracking)
IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
ENDIF ()

message("Build type: " ${CMAKE_BUILD_TYPE})


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV 4 REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable(3D_object_tracking src/camFusion_Student.cpp src/finalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries(3D_object_tracking ${OpenCV_LIBRARIES})