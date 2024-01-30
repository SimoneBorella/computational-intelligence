cmake_minimum_required(VERSION 3.16)
project(quixo)
set(CMAKE_BUILD_TYPE Release) #don't compile debug stuff

# Set max optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

message(STATUS "${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp)

