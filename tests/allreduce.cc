#include <linalg.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "testing.hh"

const int ntest = 5;

template <typename T>
int test(int64_t m, int64_t n, int64_t stride, Linalg::Comm &c, const Linalg::Place loc) {
    auto A = std::make_shared<Linalg::Tile<T> >(m, n, stride, loc);
    std::vector<double> results(ntest);
    c.ctxt->set<T>(A, 1.0);
    c.ctxt->sync();

    for(int i=0; i<ntest; i++) {
        double time = omp_get_wtime();
        c.allreduce_sum(A, A);
        c.ctxt->sync();
        time = omp_get_wtime() - time;
        results[i] = time;
        if(c.rank == 0)
            printf("allreduce time = %f sec., MB/s = %f\n", time, sizeof(T)*stride*n / (time*1024*1024));
    }
    if(c.rank == 0)
        print_times(results, sizeof(T)*stride*n / (1024.*1024));

    auto B = std::make_shared<Linalg::Tile<T> >(m, n, stride, Linalg::Place::Host);
    c.ctxt->set<T>(B, std::pow(c.ranks, ntest));
    Linalg::TileP<T> Ax = A;
    if(loc == Linalg::Place::CUDA) {
        Ax = std::make_shared<Linalg::Tile<T> >(m, n, stride, Linalg::Place::Host);
        c.ctxt->copy(Ax, A);
    }
    /*if(c.rank == 0)
        printf("ans = %f, expected %f\n", std::abs(Ax->at(0,0)), std::abs(B->at(0,0)));*/
    c.ctxt->sync();
    double err = nrm(Ax, B);

    return err > 1e-18;
}

int main(int argc, char *argv[]) {
    int64_t m = 500;
    int64_t n = 400;
    int64_t stride = 512;

    if(argc == 4) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        stride = atol(argv[3]);
        assert(stride >= m);
    }

    auto mpi  = Linalg::MPIH(&argc, &argv);
    auto ctxt = std::make_shared<Linalg::Context>();
    Linalg::Comm c(mpi, ctxt);

    int ret = 0;

    if(mpi.rank == 0)
        printf("Host\n");
#define call_test(value_t) ret += test<value_t>(m,n,stride,c,Linalg::Place::Host)
    instantiate_template(call_test)

    #ifdef ENABLE_CUDA
    if(mpi.rank == 0)
        printf("CUDA\n");
#define call_test2(value_t) ret += test<value_t>(m,n,stride,c,Linalg::Place::CUDA)
    instantiate_template(call_test2)
    #endif

    return ret;
}

