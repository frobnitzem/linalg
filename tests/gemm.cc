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

    auto A = std::make_shared<Linalg::Tile<T> >(m, k, m, loc);
    auto B = std::make_shared<Linalg::Tile<T> >(k, n, k, loc);
    auto C = std::make_shared<Linalg::Tile<T> >(m, n, m, loc);

    c.set<T>(A, 1.0);
    c.set<T>(B, 0.5);
    c.set<T>(C, 0.1);
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
    print_times(results, gflop);

    double ans0 = 0.1 - k*ntests*0.5;

    auto ans = std::make_shared<Linalg::Tile<T> >(m, n, roundup(m,32), Linalg::Place::Host);
    c.set<T>(ans, ans0);

    Linalg::TileP<T> Cx = C;
    if(C->loc != Linalg::Place::Host) {
        Cx  = std::make_shared<Linalg::Tile<T> >(m, n, m, Linalg::Place::Host);
        c.copy(Cx, C);
    }
    c.sync();

    double err = nrm(Cx, ans);
    bool ret = err > std::abs( ans0*max_epsilon<T>() );
    if(ret)
        printf("ans = %f, expected = %f, err = %e, max = %e\n",
                std::abs(Cx->at(0,0)), std::abs(ans->at(0,0)), err, std::abs( ans0*max_epsilon<T>()));

    return ret;
}

int main(int argc, char *argv[]) {
    Linalg::Context c;
    setup(m,n,k)
    int ret = 0;

#define call_test(value_t) ret += test<value_t>(m,n,k,c,Linalg::Place::Host)
    if(m < 1024 || n < 1024 || k < 1024) {
        printf("Host\n");
        instantiate_template(call_test)
    }

#ifdef ENABLE_CUDA
    printf("CUDA\n");
#define call_test2(value_t) ret += test<value_t>(m,n,k,c,Linalg::Place::CUDA)
    instantiate_template(call_test2)
#endif

    return ret;
}
