# Find the OpenBLAS library

# Compatible layer for CMake <3.12. OPENBLAS_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${OPENBLAS_ROOT})
find_package(OpenBLAS)

set(cblas_FOUND FALSE)
if(OpenBLAS_FOUND)
  set(cblas_FOUND TRUE)
  message (STATUS "Found OpenBLAS ${OpenBLAS_VERSION}")
  message (STATUS "include: ${OpenBLAS_INCLUDE_DIRS}, library: ${OpenBLAS_LIBRARIES}")
  # Create a new-style imported target (cblas)
  add_library(cblas SHARED IMPORTED)
  target_include_directories(cblas INTERFACE ${OpenBLAS_INCLUDE_DIRS})
  set_target_properties(
      cblas PROPERTIES
      IMPORTED_LOCATION ${OpenBLAS_LIBRARIES}
  )
  mark_as_advanced(OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
endif()
