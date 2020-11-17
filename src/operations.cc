#include <linalg.hh>

namespace Linalg {

template <typename value_t>
void Context::gemm(const value_t alpha, const TileP<value_t> A,
                   const TileP<value_t> B, const value_t beta,
                   TileP<value_t> C) {

    blas_error_if_msg(A->m != C->m || A->n != B->m || B->n != C->n,
                            "Invalid dimensions for gemm");
    blas_error_if_msg(C->loc != B->loc || C->loc != A->loc,
                            "gemm called on tiles from different locations");

    switch(C->loc) {
    case HostLoc: {
        blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                    C->m, C->n, A->n,
                    -1.0, A->data, A->stride,
                          B->data, B->stride,
                     1.0, C->data, C->stride );

    } break;
    default: {
        #ifndef ENABLE_CUDA
        assert(0);
        #endif
        blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                    C->m, C->n, A->n,
                    -1.0, A->data, A->stride,
                          B->data, B->stride,
                     1.0, C->data, C->stride, get_queue(C->loc) );
    } break;
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

