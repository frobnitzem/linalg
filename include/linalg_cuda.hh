#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif

#ifndef ENABLE_CUDA
// stubs
    #define CHECKCUDA(cmd)   assert(0);
    #define CHECKCUBLAS(cmd) assert(0);
    // remove this call if not using CUDA
    #define CHECK0CUDA(cmd)
#else // ENABLE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECKCUDA(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#define CHECK0CUDA(cmd) CHECKCUDA(cmd)

static const char* _cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}
#define CHECKCUBLAS(cmd) do {                       \
  cublasStatus_t e = cmd;                           \
  if( e != CUBLAS_STATUS_SUCCESS ) {                \
    printf("Failed: cuBLAS error %s:%d '%s'\n",     \
      __FILE__,__LINE__,_cublasGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#endif // ENABLE_CUDA
