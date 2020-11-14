cmake_minimum_required(VERSION 3.17)

#####################################################################
# ATTENTION: Replace languages with your own list.                  #
project(linalg VERSION 1.0 LANGUAGES CXX)
#####################################################################

# Global Options
add_library(linalg_compiler_flags INTERFACE)
#####################################################################
# ATTENTION: add all relevant compiler flags here as cmake features #
# see https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#manual:cmake-compile-features(7)
# note: INTERFACE is correct here, since 'linalg_compiler_flags' is a #
# "target", so things that depend on it see only its "public"       #
# and "interface" properties.                                       #
target_compile_features(linalg_compiler_flags INTERFACE cxx_std_11)
#####################################################################

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_TESTS "Build the tests accompanying this library." ON)
configure_file(include/linalg.hh.in include/linalg.hh)


#####################################################################
# ATTENTION: add all your library's source files to this list
add_library(linalg src/linalg.cc)
#####################################################################

# library code properties
target_link_libraries(linalg linalg_compiler_flags)
target_include_directories(linalg
                           PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                            $<INSTALL_INTERFACE:include>
                           )
# lib needs PIC when BUILD_SHARED_LIBS=ON
set_target_properties(linalg PROPERTIES
                      POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                      )

#add_subdirectory(example)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

#####################################################################
# ATTENTION: add all library targets you want to install here:      #
set(installable_libs linalg linalg_compiler_flags)
#####################################################################

# Attach these libraries to the LinalgTargets list of exported libs.
install(TARGETS ${installable_libs}
        DESTINATION lib
        EXPORT LinalgTargets)

#####################################################################
# ATTENTION: add all your library's installed headers to this list  #
install(FILES "${PROJECT_BINARY_DIR}/include/linalg.hh" DESTINATION include)
#install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/util.hh" DESTINATION include)
#####################################################################

# Note: we choose the following location for cmake dependency info:
# <prefix>/lib/cmake/Linalg/
# install the targets to export
install(EXPORT LinalgTargets
  FILE LinalgTargets.cmake
  NAMESPACE Linalg::
  DESTINATION lib/cmake/Linalg
)

# Create a config helper so others can call find_package(Linalg::linalg)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LinalgConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/Linalg"
  NO_SET_AND_CHECK_MACRO
  )
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LinalgConfigVersion.cmake"
  VERSION "${linalg_VERSION_MAJOR}.${linalg_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)
# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/LinalgConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/LinalgConfigVersion.cmake
  DESTINATION lib/cmake/Linalg
  )
# allow this project to be used from the build directory (before install)
#export(EXPORT LinalgTargets
#  FILE "${CMAKE_CURRENT_BINARY_DIR}/LinalgTargets.cmake"
#)
