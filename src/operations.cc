#include <linalg.hh>

namespace Linalg {

template<typename value_t> inline void
    callHostgemm(const value_t alpha, const TileP<value_t> A,
                 const TileP<value_t> B, const value_t beta, TileP<value_t> C);
template<typename value_t> inline void
    callCUDAgemm(CUBLAS_HANDLE_T, const value_t alpha, const TileP<value_t> A,
                 const TileP<value_t> B, const value_t beta, TileP<value_t> C);

#define instHostgemm(value_t, cast, hostCall) \
    template<> inline void \
        callHostgemm(const value_t alpha, const TileP<value_t> A, \
             const TileP<value_t> B, const value_t beta, TileP<value_t> C) { \
        assert(0); \
}
/*    BLASFUNC(hostCall)("N", "N", C->m,C->n,A->n, \
             cast(&alpha), cast(A->data),A->lda, \
             cast(B->data),B->lda, cast(&beta), cast(C->data),C->lda); \
}*/
// CUBLAS_OP_N,T,C
#define instCUDAgemm(value_t, cast, ccast, cudaCall) \
    template<> inline void \
        callCUDAgemm(CUBLAS_HANDLE_T handle, const value_t alpha, const TileP<value_t> A, \
             const TileP<value_t> B, const value_t beta, TileP<value_t> C) { \
    CHECKCUBLAS( cudaCall( handle, \
         CUBLAS_OP_N, CUBLAS_OP_N, C->m,C->n,A->n, \
         ccast(&alpha), ccast(A->data),A->lda, \
         ccast(B->data),B->lda, ccast(&beta), \
         cast(C->data),C->lda) ); \
}

instHostgemm(float, (float *),   sgemm)
instHostgemm(double, (double *), dgemm)
instHostgemm(std::complex<float>, reinterpret_cast<void *>, cgemm)
instHostgemm(std::complex<double>, reinterpret_cast<void *>, zgemm)
instCUDAgemm(float,  (float *), (const float *),  cublasSgemm)
instCUDAgemm(double, (double *), (const double *), cublasDgemm)
instCUDAgemm(std::complex<float>, (cuComplex *) reinterpret_cast<float *>,
                                  (const cuComplex *) reinterpret_cast<const float *>,  cublasCgemm)
instCUDAgemm(std::complex<double>, (cuDoubleComplex *) reinterpret_cast<double *>,
                                   (const cuDoubleComplex *) reinterpret_cast<const double *>, cublasZgemm)

// expands to, e.g.:
// cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C->m,C->n,A->n,
//             alpha, A->data,A->lda, B->data,B->lda, beta, C->data,C->lda);
template <typename value_t>
void Context::gemm(const value_t alpha, const TileP<value_t> A,
                   const TileP<value_t> B, const value_t beta,
                   TileP<value_t> C) {
    if(C->loc != B->loc || C->loc != A->loc) {
        printf("Error! gemm called on tiles from different locations!\n");
        return;
    }
    if(C->m != A->m || C->n != B->n || A->n != B->m) {
        printf("Error! invalid dimensions for gemm!\n");
        return;
    }
    switch(C->loc) {
    case Place::Host: {
        callHostgemm(alpha, A, B, beta, C);
    } break;
    case Place::CUDA: {
        callCUDAgemm(get_cublas_handle(), alpha, A, B, beta, C);
    } break;
    default:
        printf("Error! gemm on location %d is not implemented!\n", (int)C->loc);
        break;
    }
}

#define inst_gemm(value_t) template void Context::gemm<value_t>( \
          const value_t alpha, const TileP<value_t> A, \
          const TileP<value_t> B, const value_t beta, TileP<value_t> C)
inst_gemm(float);
inst_gemm(double);
inst_gemm(std::complex<float>);
inst_gemm(std::complex<double>);

}
