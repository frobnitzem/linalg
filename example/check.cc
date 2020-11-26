#include <linalg.hh>

/*    <-- ntile -->
 * ^  1   1   1   1
 * |
 *    1   1   1   1
 * m
 * t  2   2   2   2
 * i
 * l  2   2   2   2
 * e
 *    3   3   3   3
 * |
 * v  3   3   3   3
 *
 */

Matrix mkMat() {
    std::vector<int64_t> rows();
    std::vector<int64_t> cols();
    // ea. processor owns contiguous rows
    std::function<int64_t(int64_t,int64_t)> inTileMb = [=](int64_t m) { return m*p/M; };
    // ea. processor owns all columns
    std::function<int64_t(int64_t)> inTileMb = [=](int64_t m) { return m*p/M; };

    Matrix(m, n,
           const std::function<int64_t(int64_t)> inTileMb,
           const std::function<int64_t(int64_t)> inTileNb,
           std::shared_ptr<CartGroup> cart);
}

template <typename T>
int test(int64_t m, int64_t n, int64_t stride, Linalg::Comm &c, const Linalg::Place loc) {

    auto X = std::make_shared<Linalg::Tile<T> >(m, n, stride, loc);
    auto S = std::make_shared<Linalg::Tile<T> >(m, n, stride, loc);

    std::vector<double> results(ntest);
    c.ctxt->set<T>(A, 1.0);
    c.ctxt->sync();

    for(int i=0; i<ntest; i++) {
        double time = omp_get_wtime();
        c.allreduce_sum(A, A);
        c.ctxt->sync();
        time = omp_get_wtime() - time;
        results[i] = time;
        if(c.mpi->rank == 0)
            printf("allreduce time = %f sec., MB/s = %f\n", time, sizeof(T)*stride*n / (time*1024*1024));
    }
    if(c.mpi->rank == 0)
        print_times(results, sizeof(T)*stride*n / (1024.*1024));


    auto B = std::make_shared<Linalg::Tile<T> >(m, n, stride, loc);
    c.ctxt->set<T>(B, std::pow(c.mpi->ranks, ntest));
    Linalg::TileP<T> Ax = A;
    if(loc == Linalg::Place::CUDA) {
        Ax = std::make_shared<Linalg::Tile<T> >(m, n, stride, Linalg::Place::Host);
        c.ctxt->copy(Ax, A);
    }
    /*if(c.mpi->rank == 0)
        printf("ans = %f, expected %f\n", std::abs(Ax->at(0,0)), std::abs(B->at(0,0)));*/
    c.ctxt->sync();
    double err = nrm(Ax, B);

    return err > 1e-18;
}

int main(int argc, char *argv[]) {
    //int64_t m = 100000; // 4GB of doubles
    //int64_t n = 5000;
    int64_t m = 10000; // 4GB of doubles
    int64_t n = 500;
    int64_t stride = 512;
    int p = 2, q = 2;

    if(argc == 4) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        stride = atol(argv[3]);
        assert(stride >= m);
    }

    auto mpi = std::make_shared<Linalg::MPIH>(&argc, &argv);
    if(mpi->ranks < p*q) {
        printf("ERROR: Run this with >= %d MPI ranks.\n", p*q);
        return 1;
    }
    auto ctxt = std::make_shared<Linalg::Context>();
    Linalg::Comm c(mpi, ctxt);

    int ret = 0;

    if(mpi->rank == 0)
        printf("Host\n");
#define call_test(value_t) ret += test<value_t>(m,n,stride,c,Linalg::Place::Host)
    instantiate_template(call_test)

    #ifdef ENABLE_CUDA
    if(mpi->rank == 0)
        printf("CUDA\n");
#define call_test2(value_t) ret += test<value_t>(m,n,stride,c,Linalg::Place::CUDA)
    instantiate_template(call_test2)
    #endif

    return ret;
}
