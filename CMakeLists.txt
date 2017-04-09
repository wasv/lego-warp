cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( squares )
find_package( OpenCV REQUIRED )
add_executable( squares squares.cpp )
target_link_libraries( squares ${OpenCV_LIBS} )

add_executable( contours contours.cpp )
target_link_libraries( contours ${OpenCV_LIBS} )

add_executable( lego-detect lego-detect.cpp )
target_link_libraries( lego-detect ${OpenCV_LIBS} )
