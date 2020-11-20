#include <linalg.hh>
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

    int m = 500;
    int n = 400;
    int stride = 512;

    auto mpi  = std::make_shared<Linalg::MPIH>(&argc, &argv);
    auto ctxt = std::make_shared<Linalg::Context>();
    Linalg::Comm c(mpi, ctxt);

    auto A = std::make_shared<Linalg::Tile<T> >(m, n, stride, loc);
    ctxt->set<T>(A, 1.0);
    ctxt->sync();

    for(int i=0; i<5; i++) {
        double time = omp_get_wtime();
        c.allreduce_sum(A, A);
        ctxt->sync();
        time = omp_get_wtime() - time;
        printf("allreduce time = %f sec., MB/s = %f\n", time, sizeof(T)*stride*n / (time*1024*1024));
    }

    return 0;
}

