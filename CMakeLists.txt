cmake_minimum_required(VERSION 3.18)
#LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

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
target_compile_features(linalg_compiler_flags INTERFACE cxx_std_14)
#####################################################################

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(DISABLE_CUDA "Don't use cuda backend, regardless of blaspp support." OFF)
option(BUILD_TESTS "Build the tests accompanying this library." ON)

#####################################################################
# ATTENTION: add all your library's static exported header files to this list
set(LINALG_HEADERS
    include/linalg_cuda.hh
    include/linalg.hh
    include/matrix.hh
    include/operations.hh)
#####################################################################

#####################################################################
# ATTENTION: add all your library's source files to this list
set(LINALG_SOURCES
    src/linalg.cc
    src/matrix.cc
    src/operations.cc)
set(LINALG_CUDA_SOURCES
    src/set_cuda.cu)
#####################################################################

add_library(linalg ${LINALG_SOURCES})

find_package(OpenMP   3.1        REQUIRED)
find_package(MPI      1.0        REQUIRED)
find_package(blaspp   2020.10.02 REQUIRED)
get_target_property(blaspp_interface_compile_definitions blaspp INTERFACE_COMPILE_DEFINITIONS)
if(NOT DISABLE_CUDA AND "${blaspp_interface_compile_definitions}" MATCHES "BLASPP_WITH_CUBLAS")
    message(STATUS "Enabling CUDA (detected BLASPP_WITH_CUBLAS).")
    set(ENABLE_CUDA ON)
else()
    set(ENABLE_CUDA OFF)
endif()
if(ENABLE_CUDA)
    enable_language(CUDA)
    set_target_properties(linalg PROPERTIES
                          CUDA_STANDARD 14
                          CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET linalg PROPERTY CUDA_ARCHITECTURES 35 50 70 72)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(linalg CUDA::cudart CUDA::cublas)

    target_sources(linalg PRIVATE ${LINALG_CUDA_SOURCES})
endif()

# library code properties
target_link_libraries(linalg linalg_compiler_flags blaspp MPI::MPI_CXX OpenMP::OpenMP_CXX)
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

configure_file(include/linalg_config.hh.in include/linalg_config.hh)
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
install(FILES "${PROJECT_BINARY_DIR}/include/linalg_config.hh" DESTINATION include)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/linalg.hh"  DESTINATION include)
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
