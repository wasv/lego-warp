cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( lego-detect )
find_package( OpenCV REQUIRED )
add_executable( squares src/squares.cpp )
target_link_libraries( squares ${OpenCV_LIBS} )

add_executable( contours src/contours.cpp )
target_link_libraries( contours ${OpenCV_LIBS} )

add_executable( lego-detect src/lego-detect.cpp )
target_link_libraries( lego-detect ${OpenCV_LIBS} )

add_executable( sched-read src/sched-read )
target_link_libraries( sched-read ${OpenCV_LIBS} )
