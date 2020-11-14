#include <linalg.hh>

namespace Linalg {

Context::Context() : cublas_handle(CUBLAS_THREADS) {
    #ifdef ENABLE_CUDA
    // TODO: bind threads to GPU-s
    for(int i=0; i<CUBLAS_THREADS; i++) {
        CHECKCUDA(cudaStreamCreateWithFlags(&cuda_stream[i], cudaStreamNonBlocking));
        CHECKCUBLAS(cublasCreate(&cublas_handle[i]));
        CHECKCUBLAS(cublasSetStream(cublas_handle[i], cuda_stream[i]));
    }
    #endif
}
Context::~Context() {
    #ifdef ENABLE_CUDA
    for(int i=0; i<CUBLAS_THREADS; i++) {
        CHECKCUBLAS(cublasDestroy(cublas_handle[i]));
        CHECKCUDA(cudaStreamDestroy(cuda_stream[i]));
    }
    #endif
}



}
