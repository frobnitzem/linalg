#include <linalg.hh>
#include <utility>

namespace Linalg {

/* TileP functions */
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
                    alpha, A->data, A->stride,
                           B->data, B->stride,
                    beta,  C->data, C->stride );

    } break;
    default: {
        #ifndef ENABLE_CUDA
        assert(0);
        #endif
        blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                    C->m, C->n, A->n,
                    alpha, A->data, A->stride,
                           B->data, B->stride,
                    beta,  C->data, C->stride, get_queue(C->loc) );
    } break;
    }
}

#define inst_gemm(value_t) template void Context::gemm<value_t>( \
          const value_t alpha, const TileP<value_t> A, \
          const TileP<value_t> B, const value_t beta, TileP<value_t> C)
instantiate_template(inst_gemm)

template <typename value_t>
void Context::set(TileP<value_t> t, const value_t a) {
    switch(t->loc) {
    case HostLoc: {
        #pragma omp parallel for collapse(2)
        for(int64_t j=0; j<t->n; j++) {
            for(int64_t i=0; i<t->m; i++) {
                t->data[j*t->stride+i] = a;
            }
        }
    } break;
    default: {
        #ifdef ENABLE_CUDA
        set_cuda(t, a);
        #else
        assert(0);
        #endif
    } break;
    }
}
#define inst_set(value_t) template void Context::set<value_t>( \
          const TileP<value_t>, const value_t)
instantiate_template(inst_set)

/* TileView functions */
template <typename value_t>
void Context::gemm(const value_t alpha, const TileView<value_t> A,
                   const TileView<value_t> B, const value_t beta,
                   TileView<value_t> C) {

    blas_error_if_msg(A.mb() != C.mb() || A.nb() != B.mb() || B.nb() != C.nb(),
                            "Invalid dimensions for gemm");
    blas_error_if_msg(C.device() != B.device() || C.device() != A.device(),
                            "gemm called on tiles from different locations");

    TileView<value_t> Ap(A);
    TileView<value_t> Bp(B);
    switch(C.op()) {
    case blas::Op::NoTrans: break;
    case blas::Op::Trans: {
        Ap.trans();
        Bp.trans();
        std::swap(Ap, Bp);
    } break;
    case blas::Op::ConjTrans: {
        Ap.conjTrans();
        Bp.conjTrans();
        std::swap(Ap, Bp);
    } break;
    }

    switch(C.device()) {
    case HostLoc: {
        blas::gemm( blas::Layout::ColMajor, Ap.op(), Bp.op(),
                    Ap.mb(), Bp.nb(), Ap.nb(),
                    alpha, Ap.data(), Ap.stride(),
                           Bp.data(), Bp.stride(),
                     beta, C.data(), C.stride() );

    } break;
    default: {
        #ifndef ENABLE_CUDA
        assert(0);
        #endif
        blas::gemm( blas::Layout::ColMajor, Ap.op(), Bp.op(),
                    Ap.mb(), Bp.nb(), Ap.nb(),
                    alpha, Ap.data(), Ap.stride(),
                           Bp.data(), Bp.stride(),
                     beta, C.data(), C.stride(), get_queue(C.device()) );
    } break;
    }
}

#define inst_gemmV(value_t) template void Context::gemm<value_t>( \
          const value_t alpha, const TileView<value_t> A, \
          const TileView<value_t> B, const value_t beta, TileView<value_t> C)
instantiate_template(inst_gemmV)
}

