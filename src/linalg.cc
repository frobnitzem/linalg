#include <linalg.hh>

namespace Linalg {

Context::Context() : queue(CUBLAS_THREADS) {
    #ifdef ENABLE_CUDA
    blas::Device device = 0;
    // TODO: set queue devices explicitly
    #endif
}

}
