enable_testing()

# define a function to simplify adding tests
function(do_test target)
  add_executable(${target} ${target}.cc)
  target_link_libraries(${target} PUBLIC linalg)
  add_test(NAME ${target} COMMAND ${target})
endfunction(do_test)

function(mat_test target)
  add_executable(${target} ${target}.cc)
  target_link_libraries(${target} PUBLIC linalg)
  add_test(NAME ${target} COMMAND ${target} 200 500 100)
endfunction(mat_test)

do_test(checklinalg)
do_test(checkblas)
mat_test(gemm)
mat_test(gemmV)
do_test(copy)
do_test(allreduce)

if(ENABLE_CUDA)
  do_test(checkblas_device)
endif()
