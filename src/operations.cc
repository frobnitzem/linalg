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
    case Place::Host: {
        blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                    C->m, C->n, A->n,
                    alpha, A->data, A->stride,
                           B->data, B->stride,
                    beta,  C->data, C->stride );

    } break;
    case Place::CUDA: {
        #ifndef ENABLE_CUDA
        assert(0);
        #endif
        blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                    C->m, C->n, A->n,
                    alpha, A->data, A->stride,
                           B->data, B->stride,
                    beta,  C->data, C->stride, get_queue() );
    } break;
    default: assert(0);
    }
}

#define inst_gemm(value_t) template void Context::gemm<value_t>( \
          const value_t alpha, const TileP<value_t> A, \
          const TileP<value_t> B, const value_t beta, TileP<value_t> C)
instantiate_template(inst_gemm)

template <typename value_t>
void Context::set(TileP<value_t> t, const value_t a) {
    switch(t->loc) {
    case Place::Host: {
        #pragma omp parallel for collapse(2)
        for(int64_t j=0; j<t->n; j++) {
            for(int64_t i=0; i<t->m; i++) {
                t->data[j*t->stride+i] = a;
            }
        }
    } break;
    case Place::CUDA: {
        #ifndef ENABLE_CUDA
        assert(0);
        #else
        set_cuda(t, a);
        #endif
    } break;
    default: assert(0);
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

    assert(A.mb() == C.mb());
    assert(A.nb() == B.mb());
    assert(B.nb() == C.nb());
    blas_error_if_msg(A.mb() != C.mb() || A.nb() != B.mb() || B.nb() != C.nb(),
                            "Invalid dimensions for gemm");
    blas_error_if_msg(C.device() != B.device() || C.device() != A.device(),
                            "gemm called on tiles from different locations");

    TileView<value_t> Ap(A);
    TileView<value_t> Bp(B);
    value_t a = alpha;
    value_t b = beta;
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
        a = is_complex_t<value_t>::conj(a);
        b = is_complex_t<value_t>::conj(b);
    } break;
    default: assert(0);
    }

    switch(C.device()) {
    case Place::Host: {
        blas::gemm( blas::Layout::ColMajor, Ap.op(), Bp.op(),
                    Ap.mb(), Bp.nb(), Ap.nb(),
                        a, Ap.data(), Ap.stride(),
                           Bp.data(), Bp.stride(),
                        b, C.data(),  C.stride() );

    } break;
    case Place::CUDA: {
        #ifndef ENABLE_CUDA
        assert(0);
        #endif
        blas::gemm( blas::Layout::ColMajor, Ap.op(), Bp.op(),
                    Ap.mb(), Bp.nb(), Ap.nb(),
                        a, Ap.data(), Ap.stride(),
                           Bp.data(), Bp.stride(),
                        b, C.data(),  C.stride(), get_queue() );
    } break;
    default: assert(0);
    }
}
#define inst_gemmV(value_t) template void Context::gemm<value_t>( \
          const value_t alpha, const TileView<value_t> A, \
          const TileView<value_t> B, const value_t beta, TileView<value_t> C)
instantiate_template(inst_gemmV)

/* This handles copying between devices.
 * It requires identical types and tile shapes.
 * It allows unequal column-strides.
 */
template <typename value_t>
void Context::copy(TileP<value_t> dst, TileP<value_t>src) {
    blas_error_if_msg(dst->m != src->m || dst->n != src->n,
                      "copy requires identical tile dimensions");
    switch(dst->loc) {
    case Place::Host: {
        switch(src->loc) {
        case Place::Host: {
            #pragma omp parallel for collapse(2)
            for(int64_t j=0; j<dst->n; j++) {
                for(int64_t i=0; i<dst->m; i++) {
                    dst->at(i,j) = src->at(i,j);
                }
            }
        } break;
        case Place::CUDA: {
            blas::device_getmatrix(dst->m, dst->n, src->data, src->stride,
                                   dst->data, dst->stride, get_queue());
        } break;
        default: assert(0);
        }
    } break;
    case Place::CUDA: {
        switch(src->loc) {
        case Place::Host: {
            blas::device_setmatrix(dst->m, dst->n, src->data, src->stride,
                                   dst->data, dst->stride, get_queue());
        } break;
        case Place::CUDA: {
            #ifndef ENABLE_CUDA
            assert(0);
            #else
            copy_cuda<value_t, value_t>(TileView<value_t>(dst), TileView<value_t>(src));
            #endif
        } break;
        default: assert(0);
        }
    } break;
    default: assert(0);
    }
}
#define inst_copy(value_t) template void Context::copy<value_t>( \
                            TileP<value_t> dst, TileP<value_t>src);
instantiate_template(inst_copy)

/* This handles tile transposition and changing strides,
 * but requires identical devices. */
template <typename dst_t, typename src_t>
void Context::copy(TileView<dst_t> dst, const TileView<src_t> src) {
    blas_error_if_msg(dst.mb() != src.mb() || dst.nb() != src.nb(),
                      "copy requires identical tile dimensions");
    blas_error_if_msg(dst.device() != src.device(),
                      "copy called on tiles from different locations");
    blas_error_if_msg( (dst.op() == blas::Op::ConjTrans) != (src.op() == blas::Op::ConjTrans),
                      "copy can't yet do complex conjugate");
    blas_error_if_msg(dst.uplo() != blas::Uplo::General || src.uplo() != blas::Uplo::General,
                      "copy can't yet do uplo != General");

    switch(dst.device()) {
    case Place::Host: {
        // same transpose status
        if( (src.op() == blas::Op::NoTrans) == (dst.op() == blas::Op::NoTrans) ) {
            #pragma omp parallel for collapse(2)
            for(int j=0; j<dst.t->n; j++) {
                for(int i=0; i<dst.t->m; i++) {
                        dst.t->at(i,j) = src.t->at(i,j);
                }
            }
        } else { // transpose-copy
            #pragma omp parallel for collapse(2)
            for(int j=0; j<dst.t->n; j++) {
                for(int i=0; i<dst.t->m; i++) {
                        dst.t->at(i,j) = src.t->at(j,i);
                }
            }
        }
    } break;
    case Place::CUDA: {
        #ifndef ENABLE_CUDA
        assert(0);
        #else
        copy_cuda<dst_t, src_t>(dst, src);
        #endif
    } break;
    default: assert(0);
    }
}
#define inst_copyV2(dst_t, src_t) template void Context::copy<dst_t, src_t>( \
                          TileView<dst_t>, const TileView<src_t>)
#define inst_copyV(dst_t) inst_copyV2(dst_t, float); \
                          inst_copyV2(dst_t, double)
instantiate_template(inst_copyV)
inst_copyV2(std::complex<float>,  std::complex<float>);
inst_copyV2(std::complex<float>,  std::complex<double>);
inst_copyV2(std::complex<double>, std::complex<float>);
inst_copyV2(std::complex<double>, std::complex<double>);

}

