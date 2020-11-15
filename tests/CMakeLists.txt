enable_testing()

add_executable(checklinalg checklinalg.cc)
target_link_libraries(checklinalg PUBLIC linalg)
add_test(NAME checklinalg COMMAND checklinalg)

add_executable(checkblas checkblas.cc)
target_link_libraries(checkblas PUBLIC linalg)
add_test(NAME checkblas COMMAND checkblas)

if(ENABLE_CUDA)
  add_executable(checkblas_device checkblas_device.cc)
  target_link_libraries(checkblas_device PUBLIC linalg)
  add_test(NAME checkblas_device COMMAND checkblas_device)

  add_executable(gemm gemm.cc)
  target_link_libraries(gemm PUBLIC linalg)
  add_test(NAME gemm COMMAND gemm)
endif()
