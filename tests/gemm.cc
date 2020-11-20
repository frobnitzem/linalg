#include <linalg.hh>
#include <blas/flops.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <bits/stdc++.h>

#ifdef ENABLE_CUDA
const Linalg::Place loc = Linalg::Place::CUDA;
#else
const Linalg::Place loc = Linalg::Place::Host;
#endif

int main(int argc, char *argv[]) {
    using T = float;
    std::vector<double> results(5);
    const int ntests = 5;

    if(argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }
    int m = atol(argv[1]);
    int n = atol(argv[2]);
    int k = atol(argv[3]);

    Linalg::Context c;
    //blas::Queue q;
    auto A = std::make_shared<Linalg::Tile<T> >(m, k, m, loc);
    auto B = std::make_shared<Linalg::Tile<T> >(k, n, k, loc);
    auto C = std::make_shared<Linalg::Tile<T> >(m, n, m, loc);
    c.set<T>(A, 1.0);
    c.set<T>(B, 0.5);
    c.set<T>(C, 0.0);
    c.sync();

    double gflop = blas::Gflop <T>::gemm( m, n, k );
    for(int i=0; i<ntests; i++) {
        double time = omp_get_wtime();
        c.gemm<T>(-1.0, A, B, 1.0, C);
        c.sync();
        time = omp_get_wtime() - time;
        printf("GEMM time = %f sec.   gflops = %f\n", time, gflop / time);
        results[i] = time;
    }
    //blas::device_getmatrix(m, n, dC, ldc, C.data(), ldc, queue);
    //queue.sync();

    sort(results.begin(), results.end());
    double sum = 0.0;
    int N = 0;
    // average of fastest times
    for(int i=0; i < (ntests+1)/2; i++) {
        sum += results[i];
        N++;
    }
    printf("GFLOPS: %f\n", N*gflop/sum);

    return 0;
}
