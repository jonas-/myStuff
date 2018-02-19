
cmake_minimum_required(VERSION 2.8.3)

project(ORBVoci)

set(ROS_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Debug)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    cv_bridge
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(DBoW2ori REQUIRED PATHS)
find_package(DLib REQUIRED)

include_directories(
  include
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  #/home/jonas/git/opengv/include
)

# ------------------------------------------------------------------------------

add_executable(voci src/createORBVoci.cpp src/ORBextractor.cc)

target_link_libraries(voci
    ${DBoW2ori_LIBS}
    ${DLib_LIBS}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
)
