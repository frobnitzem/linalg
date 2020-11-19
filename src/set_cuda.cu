#include <linalg.hh>

template <typename T>
__global__ void set_cuda_kernel(size_t n, T a, T *x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
          x[i] = a;
      }
}

namespace Linalg {
template <typename T>
void Context::set_cuda(TileP<T> t, const T a) {
    int device = t->loc;
    assert(device >= 0); // must be gpu-resident
    blas::set_device(device);
    if(a == (T)0.0) {
        CHECKCUDA( cudaMemset(t->data, 0, t->stride*t->n*sizeof(T)) );
        return;
    }

    int numSMs; // the V100 has 160 SMs, each with 32 "CUDA cores"
    CHECKCUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device) );
    numSMs *= 32;
    size_t N = (size_t)t->stride * t->n;
    size_t maxblk = (N+31)/32;
    if(maxblk < numSMs) numSMs = maxblk;

    cudaStream_t stream = get_queue().stream();
    set_cuda_kernel<<<numSMs, 32, 0, stream >>>(N, a, t->data);
}
#define inst_set_cuda(T) template void Context::set_cuda(TileP<T> t, const T a)
instantiate_template(inst_set_cuda)
}
