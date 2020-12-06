#include <linalg.hh>

template <typename T>
static __global__ void set_kernel(size_t n, T a, T *x) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
          x[i] = a;
    }
}

template <typename T, typename U>
static __device__ T make(const U x) {
    return (T)x;
}
template<> __device__ __host__ cuComplex make(const cuComplex x) { return x; }
template<> __device__ __host__ cuDoubleComplex make(const cuDoubleComplex x) { return x; }
template<> __device__ __host__ cuComplex make(const cuDoubleComplex x) { return {(float)x.x, (float)x.y}; }
template<> __device__ __host__ cuDoubleComplex make(const cuComplex x) { return {x.x,x.y}; }
template<> __device__ __host__ cuComplex make(const float x) { return cuComplex{x,0.0}; }
template<> __device__ __host__ cuDoubleComplex make(const float x) { return cuDoubleComplex{x,0.0}; }
template<> __device__ __host__ cuComplex make(const double x) { return cuComplex{(float)x,0.0}; }
template<> __device__ __host__ cuDoubleComplex make(const double x) { return cuDoubleComplex{x,0.0}; }

// copy. A and B are mxn.
template <typename T, typename U>
static __global__ void copy_kernel(size_t m, size_t n,
                                              T *A, size_t lda,
                                        const U *B, size_t ldb) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n;
             j += blockDim.y * gridDim.y) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
                 i += blockDim.x * gridDim.x) {
            A[i+j*lda] = make<T,U>(B[i+j*ldb]);
        }
    }
}

// copy with transposition. A is mxn, B is nxm
// TODO: optimize this by pre-fetching B
template <typename T, typename U>
static __global__ void copy_tr_kernel(size_t m, size_t n,
                                              T *A, size_t lda,
                                        const U *B, size_t ldb) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n;
             j += blockDim.y * gridDim.y) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
                 i += blockDim.x * gridDim.x) {
            A[i+j*lda] = make<T,U>(B[j+i*ldb]);
        }
    }
}

static int num_blks(size_t N, int blk) {
    //int numSMs; // the V100 has 160 SMs, each with 32 "CUDA cores"
    //CHECKCUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device) );
    //numSMs *= 32;
    int numSMs = 160*32;
    size_t maxblk = (N+blk-1) / blk; // maximum block size
    return maxblk < numSMs ? maxblk : numSMs;
}

namespace Linalg {
/* Casting rules for cuda datatypes */
template <typename value_t> struct CUDA_T {};
template <> struct CUDA_T<float>  {
    static const float *cast(const float *x) { return x; }
    static       float *cast(      float *x) { return x; }
};
template <> struct CUDA_T<double>  {
    static const double *cast(const double *x) { return x; }
    static       double *cast(      double *x) { return x; }
};
template <> struct CUDA_T<std::complex<float>>  {
    static const cuComplex *cast(const std::complex<float> *x)
        { return (const cuComplex *)reinterpret_cast<const float *>(x); }
    static      cuComplex *cast(      std::complex<float> *x)
        { return (cuComplex *)reinterpret_cast<      float *>(x); }
};
template <> struct CUDA_T<std::complex<double>>  {
    static const cuDoubleComplex *cast(const std::complex<double> *x)
        { return (const cuDoubleComplex *)reinterpret_cast<const double *>(x); }
    static       cuDoubleComplex *cast(      std::complex<double> *x)
        { return (cuDoubleComplex *)reinterpret_cast<      double *>(x); }
};

template <typename T>
void Context::set_cuda(TileP<T> t, const T a) {
    cudaStream_t stream = get_queue().stream();
    if(a == (T)0.0) {
        CHECKCUDA( cudaMemsetAsync(t->data, 0, t->stride*t->n*sizeof(T), stream) );
        return;
    }

    size_t N = (size_t)t->stride * t->n;
    int blks = num_blks(N, 32);

    set_kernel<<<blks, 32, 0, stream>>>(N, *CUDA_T<T>::cast(&a), CUDA_T<T>::cast(t->data));
    CHECKCUDA( cudaPeekAtLastError() );
}
#define inst_set_cuda(T) template void Context::set_cuda(TileP<T> t, const T a)
instantiate_template(inst_set_cuda)

// expands to, e.g.:
// cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C->m,C->n,A->n,
//             alpha, A->data,A->lda, B->data,B->lda, beta, C->data,C->lda);

template <typename dst_t, typename src_t>
void Context::copy_cuda(TileView<dst_t> dst, const TileView<src_t> src) {
    cudaStream_t stream = get_queue().stream();
    int blks = num_blks(dst.t->m, 32);

    // same transpose status
    if( (src.op() == blas::Op::NoTrans) == (dst.op() == blas::Op::NoTrans) ) {
        copy_kernel<<<blks, 32, 0, stream >>>(   dst.t->m, dst.t->n,
                                   //dtrans::cast(dst.data()), dst.t->stride,
                                   //strans::cast(src.data()), src.t->stride);
                                   CUDA_T<dst_t>::cast(dst.data()), dst.t->stride,
                                   CUDA_T<src_t>::cast(src.data()), src.t->stride);
    } else { // different transpose status
        copy_tr_kernel<<<blks, 32, 0, stream >>>(dst.t->m, dst.t->n,
                                   //dtrans::cast(dst.data()), dst.t->stride,
                                   //strans::cast(src.data()), src.t->stride);
                                   CUDA_T<dst_t>::cast(dst.data()), dst.t->stride,
                                   CUDA_T<src_t>::cast(src.data()), src.t->stride);
    }
}
#define inst_copy_cudaV2(dst_t, src_t) template void Context::copy_cuda<dst_t, src_t>( \
                          TileView<dst_t>, const TileView<src_t>)
#define inst_copy_cudaV(dst_t) inst_copy_cudaV2(dst_t, float); \
                          inst_copy_cudaV2(dst_t, double)
instantiate_template(inst_copy_cudaV)
inst_copy_cudaV2(std::complex<float>, std::complex<float>);
inst_copy_cudaV2(std::complex<float>, std::complex<double>);
inst_copy_cudaV2(std::complex<double>, std::complex<float>);
inst_copy_cudaV2(std::complex<double>, std::complex<double>);

}
