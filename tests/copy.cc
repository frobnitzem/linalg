// Test TileView copy.
#include <linalg.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#ifdef ENABLE_CUDA
const Linalg::Place loc = Linalg::Place::CUDA;
#else
const Linalg::Place loc = Linalg::Place::Host;
#endif

// copy a strided tile to a dense one
int main(int argc, char *argv[]) {
    using T = float;

    int m = 500;
    int n = 400;
    int stride1 = 512;
    int stride2 = m;

    Linalg::Context c;
    auto At = std::make_shared<Linalg::Tile<T> >(m, n, stride1, loc);
    auto Bt = std::make_shared<Linalg::Tile<T> >(m, n, stride2, loc);
    auto A = Linalg::TileView<T>(At);
    auto B = Linalg::TileView<T>(Bt);
    c.set<T>(A, 1.0);
    c.sync();

    for(int i=0; i<5; i++) {
        double time = omp_get_wtime();
        c.copy(B, A);
        c.sync();
        time = omp_get_wtime() - time;
        printf("copy time = %f sec., GB/s = %f\n", time, sizeof(T)*m*n / time);
    }
    //blas::device_getmatrix(m, n, dC, ldc, C.data(), ldc, queue);
    //c.sync();

    return 0;
}

