#include <blas.hh>

#include <vector>
#include <stdio.h>
#include <omp.h>
#include <blas/flops.hh>

template< typename T >
inline T roundup( T x, T y ) {
    return T( (x + y - 1) / y ) * y;
}

//------------------------------------------------------------------------------
template <typename T>
void run( int m, int n, int k, blas::Device device ) {
    int align = 1;
    int64_t lda = roundup( m, align );
    int64_t ldb = roundup( k, align );
    int64_t ldc = roundup( m, align );


    // device specifics
    blas::Queue queue(device, 0);
    std::vector<T> A( size_t(lda)*k, 1.0 );  // m-by-k
    std::vector<T> B( size_t(ldb)*n, 2.0 );  // k-by-n
    std::vector<T> C( size_t(ldc)*n, 3.0 );  // m-by-n
    T* dA = blas::device_malloc<T>(size_t(lda)*k);
    T* dB = blas::device_malloc<T>(size_t(ldb)*n);
    T* dC = blas::device_malloc<T>(size_t(ldc)*n);

    blas::device_setmatrix(m, k, A.data(), lda, dA, lda, queue); // copy to device
    blas::device_setmatrix(k, n, B.data(), ldb, dB, ldb, queue);
    blas::device_setmatrix(m, n, C.data(), ldc, dC, ldc, queue);
    queue.sync();

    // ... fill in application data into A, B, C ...

    double time = omp_get_wtime();
    if(1) {
    // C = -1.0*A*B + 1.0*C
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, n, k,
                -1.0, dA, lda,
                      dB, ldb,
                 1.0, dC, ldc, queue );
    queue.sync();
    }
    time = omp_get_wtime() - time;
    double gflop = blas::Gflop <T>::gemm( m, n, k );
    printf("GEMM time = %f sec.   gflops = %f\n", time, gflop / time);

    blas::device_getmatrix(m, n, dC, ldc, C.data(), ldc, queue);
    queue.sync();
    blas::device_free( dA );
    blas::device_free( dB );
    blas::device_free( dC );
}

//------------------------------------------------------------------------------
int main( int argc, char** argv ) {
    blas::Device device = 0;

    int m = 100, n = 200, k = 50;
    printf( "run< float >( %d, %d, %d )\n", m, n, k );
    run< float  >( m, n, k, device );

    printf( "run< double >( %d, %d, %d )\n", m, n, k );
    run< double >( m, n, k, device );

    printf( "run< complex<float> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<float>  >( m, n, k, device );

    printf( "run< complex<double> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<double> >( m, n, k, device );

    return 0;
}
