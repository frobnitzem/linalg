#include <linalg.hh>

namespace Linalg {

Context::Context() {
    #ifdef ENABLE_CUDA
    int devices;
    CHECKCUDA( cudaGetDeviceCount(&devices) );
    for(int loc=0; loc < devices; loc++) {
        queue.emplace_back((blas::Device)loc, 0);
    }
    #endif
}

}
