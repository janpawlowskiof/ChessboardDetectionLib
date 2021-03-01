cmake_minimum_required(VERSION 3.16)
project(ChessboardDetectionLib)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.4.0 REQUIRED)

#set (CMAKE_CXX_FLAGS "-I/usr/local/include/opencv4")
add_executable(ChessboardDetectionLib main.cpp chessboard_detection.h chessboard_detection.cpp)
target_link_libraries(ChessboardDetectionLib ${OpenCV_LIBS})
