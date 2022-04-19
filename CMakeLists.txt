cmake_minimum_required (VERSION 2.8)
project (AffineSkew)

find_package(OpenCV REQUIRED)

include_directories(
    ${OPENCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)

add_definitions(-std=c++11)

add_executable(main
    src/main.cpp
    src/mrogh.cpp
)
target_link_libraries(main ${OpenCV_LIBS})
