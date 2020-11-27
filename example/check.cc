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
using namespace Linalg;

template <typename T>
int test(int64_t m, int64_t n, int64_t mt, int64_t nt, CartComm &c, const Place loc) {
    int64_t mtm1 = roundup(m, mt)/mt - 1;
    int64_t ntm1 = roundup(n, nt)/nt - 1;
    // the last tile in every dimension is cut short
    auto inTileMb = [=](int64_t ti) { return ti == mtm1 ? m-mtm1*mt : mt; };
    auto inTileNb = [=](int64_t tj) { return tj == ntm1 ? n-ntm1*nt : nt; };
    auto home = [ranks = c.cart.p ](Idx idx) { return idx.first / ranks; };

    auto M = Matrix<T>(m, n, c.cart.j, loc, inTileMb, inTileNb, home, c);
    auto A = M.alloc(mt);

    const int ntest = 5;
    std::vector<double> results(ntest);
    c.ctxt->set<T>(A, 1.0);
    c.ctxt->sync();

    for(int i=0; i<ntest; i++) {
        double time = omp_get_wtime();
        //c.allreduce_sum(A, A);
        c.ctxt->sync();
        time = omp_get_wtime() - time;
        results[i] = time;
        //if(c.rank == 0)
        //    printf("allreduce time = %f sec., MB/s = %f\n", time, sizeof(T)*A->stride*A->n / (time*1024*1024));
    }
    /*if(c.rank == 0)
        print_times(results, sizeof(T)*m*n / (1024.*1024));

    auto B = std::make_shared<Tile<T> >(m, n, align, loc);
    c.ctxt->set<T>(B, std::pow(c.ranks, ntest));
    TileP<T> Ax = A;
    if(loc == Place::CUDA) {
        Ax = std::make_shared<Tile<T> >(m, n, align, Place::Host);
        c.ctxt->copy(Ax, A);
    }
    //if(c.rank == 0)
    //    printf("ans = %f, expected %f\n", std::abs(Ax->at(0,0)), std::abs(B->at(0,0)));
    c.ctxt->sync();
    double err = nrm(Ax, B);

    return err > 1e-18;*/
    return 0;
}

int main(int argc, char *argv[]) {
    //int64_t m = 100000; // 4 GB of doubles
    //int64_t n = 5000;
    int64_t m = 10000; // 40 MB of doubles
    int64_t n = 500;
    int64_t mt = 2048;
    int64_t nt = 128;
    int p = 2, q = 2;

    if(argc == 5) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        mt = atol(argv[3]);
        nt = atol(argv[3]);
    }

    MPIH      mpi(&argc, &argv);
    CartGroup cart(mpi.comm, p, q);

    if(mpi.ranks < p*q) {
        printf("ERROR: Run this with >= %d MPI ranks.\n", p*q);
        return 1;
    }
    auto ctxt = std::make_shared<Context>();
    CartComm c(cart, ctxt);

    int ret = 0;

    if(mpi.rank == 0)
        printf("Host\n");
#define call_test(value_t) ret += test<value_t>(m,n,mt,nt,c,Place::Host)
    instantiate_template(call_test)

    #ifdef ENABLE_CUDA
    if(mpi.rank == 0)
        printf("CUDA\n");
#define call_test2(value_t) ret += test<value_t>(m,n,mt,nt,c,Place::CUDA)
    instantiate_template(call_test2)
    #endif

    return ret;
}
