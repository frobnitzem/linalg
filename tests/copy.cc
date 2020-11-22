// Test TileView copy.
#include <linalg.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "testing.hh"

const int ntests = 5;

template <typename T>
int test(int64_t m, int64_t n, int64_t stride1, int64_t stride2,
                         Linalg::Context &c, const Linalg::Place loc, bool trans1, bool trans2) {
    std::vector<double> results(ntests);

    auto At = std::make_shared<Linalg::Tile<T> >(m, n, stride1, loc);
    auto Bt = std::make_shared<Linalg::Tile<T> >(m, n, stride2, loc);
    auto A = Linalg::TileView<T>(At);
    auto B = Linalg::TileView<T>(Bt);
    c.set<T>(A, 1.0);
    c.set<T>(B, 0.0);
    c.sync();

    for(int i=0; i<ntests; i++) {
        double time = omp_get_wtime();
        c.copy(B, A);
        c.sync();
        time = omp_get_wtime() - time;
        printf("copy time = %f sec., MB/s = %f\n", time, sizeof(T)*m*n / (time*1024*1024));
        results[i] = time;
    }
    print_times(results, sizeof(T)*m*n/(1024.0*1024));

    Linalg::TileP<T> Ax = At, Bx = Bt;
    if(loc == Linalg::Place::CUDA) {
        Ax = std::make_shared<Linalg::Tile<T> >(m, n, stride1, Linalg::Place::Host);
        Bx = std::make_shared<Linalg::Tile<T> >(m, n, stride2, Linalg::Place::Host);
        c.copy(Ax, At);
        c.copy(Bx, Bt);
        c.sync();
    }
    double err = nrm(Ax, Bx);

    return err > 1e-18;
}

// copy a strided tile to a dense one
int main(int argc, char *argv[]) {
    int64_t m = 500;
    int64_t n = 400;
    int64_t stride1 = 512;
    int64_t stride2 = m;
    bool tr1=false, tr2=false;
    int ret = 0;

    Linalg::Context c;

    printf("Host\n");
#define call_test(value_t) ret += test<value_t>(m,n,stride1,stride2,c,Linalg::Place::Host,tr1,tr2)
    instantiate_template(call_test)

#ifdef ENABLE_CUDA
    printf("Device\n");
#define call_test2(value_t) ret += test<value_t>(m,n,stride1,stride2,c,Linalg::Place::CUDA,tr1,tr2)
    instantiate_template(call_test2)
#endif

    return ret;
}
