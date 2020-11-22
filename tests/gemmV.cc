// Same as gemm, but using TileView-s.
#include <linalg.hh>
#include <blas/flops.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "testing.hh"

const int ntests = 5;

template <typename T>
int test(int64_t m, int64_t n, int64_t k, Linalg::Context &c, const Linalg::Place loc) {
    std::vector<double> results(ntests);

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

    double gflop = blas::Gflop <T>::gemm( m, n, k );
    for(int i=0; i<5; i++) {
        double time = omp_get_wtime();
        c.gemm<T>(-1.0, A, B, 1.0, C);
        c.sync();
        time = omp_get_wtime() - time;
        printf("GEMM time = %f sec.   gflops = %f\n", time, gflop / time);
        results[i] = time;
    }
    print_times(results, gflop);

    return 0;
}

int main(int argc, char *argv[]) {
    Linalg::Context c;
    setup(m,n,k)
    int ret = 0;

#define call_test(value_t) ret += test<value_t>(m,n,k,c,Linalg::Place::Host)
    instantiate_template(call_test)

#ifdef ENABLE_CUDA
#define call_test2(value_t) ret += test<value_t>(m,n,k,c,Linalg::Place::CUDA)
    instantiate_template(call_test2)
#endif

    return ret;
}

