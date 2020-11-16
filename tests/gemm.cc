#include <linalg.hh>
#include <blas/flops.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

int main(int argc, char *argv[]) {
    using T = float;

    if(argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }
    int m = atol(argv[1]);
    int n = atol(argv[2]);
    int k = atol(argv[3]);

    Linalg::Context c;
    //blas::Queue q;
    auto A = std::make_shared<Linalg::Tile<T> >(m, k, m, Linalg::Place::CUDA);
    auto B = std::make_shared<Linalg::Tile<T> >(k, n, k, Linalg::Place::CUDA);
    auto C = std::make_shared<Linalg::Tile<T> >(m, n, m, Linalg::Place::CUDA);

    double time = omp_get_wtime();
    c.gemm<T>(-1.0, A, B, 1.0, C);
    c.queue[0].sync();
    time = omp_get_wtime() - time;
    double gflop = blas::Gflop <T>::gemm( m, n, k );
    printf("GEMM time = %f sec.   gflops = %f\n", time, gflop / time);
    //blas::device_getmatrix(m, n, dC, ldc, C.data(), ldc, queue);
    //queue.sync();

    return 0;
}

