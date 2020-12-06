cmake_minimum_required(VERSION 3.17)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# set the project name
project(smatrix VERSION 1.0)
find_package(Linalg 1.0 REQUIRED)

add_executable(smat smat.cc)
target_link_libraries(smat PRIVATE Linalg::linalg)

set_target_properties(smat PROPERTIES
                      INSTALL_RPATH_USE_LINK_PATH YES
                      BUILD_WITH_INSTALL_RPATH YES
                      )
install(TARGETS smat DESTINATION bin)
