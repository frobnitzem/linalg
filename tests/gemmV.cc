// Same as gemm, but using TileView-s.
#include <linalg.hh>
#include <blas/flops.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#ifdef ENABLE_CUDA
const Linalg::Place loc = Linalg::Place::CUDA;
#else
const Linalg::Place loc = Linalg::Place::Host;
#endif

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
    auto At = std::make_shared<Linalg::Tile<T> >(m, k, m, loc);
    auto Bt = std::make_shared<Linalg::Tile<T> >(k, n, k, loc);
    auto Ct = std::make_shared<Linalg::Tile<T> >(m, n, m, loc);
    auto A = Linalg::TileView<T>(At);
    auto B = Linalg::TileView<T>(Bt);
    auto C = Linalg::TileView<T>(Ct);
    c.set<T>(A, 1.0);
    c.set<T>(B, 0.5);
    c.set<T>(C, 0.0);
    c.sync();

    for(int i=0; i<5; i++) {
        double time = omp_get_wtime();
        c.gemm<T>(-1.0, A, B, 1.0, C);
        c.sync();
        time = omp_get_wtime() - time;
        double gflop = blas::Gflop <T>::gemm( m, n, k );
        printf("GEMM time = %f sec.   gflops = %f\n", time, gflop / time);
    }
    //blas::device_getmatrix(m, n, dC, ldc, C.data(), ldc, queue);
    //c.sync();

    return 0;
}

