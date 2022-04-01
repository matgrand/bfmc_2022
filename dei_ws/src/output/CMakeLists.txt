cmake_minimum_required(VERSION 3.0.2)
project(output)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories( include ${catkin_INCLUDE_DIRS} )

add_executable(serialHandlerNODE_CPP 
    src/SerialHandlerCPP/app1.cpp 
    src/SerialHandlerCPP/Message.cpp 
    src/SerialHandlerCPP/ResponseHandler.cpp 
    src/SerialHandlerCPP/SerialComManager.cpp 
    src/SerialHandlerCPP/serialPortHandler.cpp)
target_link_libraries(serialHandlerNODE_CPP ${catkin_LIBRARIES})
