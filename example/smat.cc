#include <linalg.hh>
#include <math.h>
#include <utility>

using namespace Linalg;

static const int verbose = 0;
static const int progress = 0;

struct Timed {
    Timed(const std::string _name, const double _gflop, const Comm &_mpi) :
        name(_name), gflop(_gflop),
        mpi(_mpi), start(omp_get_wtime()) {}
    ~Timed() {
        if(mpi.rank != 0) return;
        mpi.ctxt->sync();
        const double time = omp_get_wtime() - start;
        printf("%s = %f sec., GFLOPS = %f\n", name.c_str(), time, gflop / time);
    }

    const std::string name;
    const double gflop;
    const Comm &mpi;
    const double start;
};

/**
 * Return a tall, skinny matrix laid
 * out contiguously across processors:
 *
 *    <-- ntile -->
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
template <typename T>
Matrix<T> newWaveFn(int64_t m, int64_t n, int64_t mbs, int64_t nbs, CartComm &c, const Place loc) {
    assert(c.cart.q == 1);

    int64_t mtm1 = roundup(m, mbs)/mbs - 1;
    int64_t ntm1 = roundup(n, nbs)/nbs - 1;
    // the last tile in every dimension is cut short
    auto inTileMb = [=](int64_t ti) { return ti == mtm1 ? m-mtm1*mbs : mbs; };
    auto inTileNb = [=](int64_t tj) { return tj == ntm1 ? n-ntm1*nbs : nbs; };
    auto home = [p = c.cart.p, mt = mtm1+1](Idx idx) { return idx.first*p/mt; };

    return Matrix<T>(m, n, c.cart.j, loc, inTileMb, inTileNb, home, c);
}

/**
 * Return a dense matrix laid out in scalapack format.
 */
template <typename T>
Matrix<T> newMatrix(int64_t m, int64_t n, int64_t mbs, int64_t nbs, CartComm &c, const Place loc) {
    int64_t mtm1 = roundup(m, mbs)/mbs - 1;
    int64_t ntm1 = roundup(n, nbs)/nbs - 1;
    // the last tile in every dimension is cut short
    auto inTileMb = [=](int64_t ti) { return ti == mtm1 ? m-mtm1*mbs : mbs; };
    auto inTileNb = [=](int64_t tj) { return tj == ntm1 ? n-ntm1*nbs : nbs; };
    auto home = [p=c.cart.p, q=c.cart.q](Idx idx) {
        int ti = idx.first  % p;
        int tj = idx.second % q;
        return tj*p + ti;
    };

    return Matrix<T>(m, n, c.cart.j, loc, inTileMb, inTileNb, home, c);
}

/**
 * Compute a p*bs x p*bs subtile of the output matrix
 * (located at ti,tj for every processor).
 *
 * The eventual output matrix has tiles bs x bs (p = q)
 * @param X = input matrix with tiles sized any x (p*bs)
 * @param out = temporary tile sized p*bs x p*bs
 *
 */
template <typename value_t>
void compute_tile(Matrix<value_t> &X, TileP<value_t> out, int64_t ti, int64_t tj) {
    Linalg::ContextP ctxt = X.mpi.ctxt;

    Timed timer("Compute Tile", 2.0*X.M*X.inTileNb(ti)*X.inTileNb(tj)/(1024.*1024*1024), X.mpi);
    ctxt->set< value_t >(out, 0.0);
    for(auto tk : X.rows) {
        auto A = X(tk,ti);
        auto B = X(tk,tj);
        A.conjTrans();
        if(verbose)
            printf("%ld %ld %ld: %ld %ld %ld gemm\n", ti, tj, tk, A.mb(), B.nb(), A.nb());

        auto tmp = slice(out, 0, A.mb(), 0, B.nb());
        ctxt->gemm< value_t >(1.0, A, B, 1.0, tmp);
    }
}

/**
 * Copy the bs x bs region at ti,tj from `out` to `S`.
 *
 * @param S = output matrix with tiles bs x bs (p = q)
 * @param out = temporary tile sized p*bs x p*bs
 *
 */
template <typename T>
void set_local(Matrix<T> &S, TileP<T> out, int64_t ti, int64_t tj) {
    if(! S.mpi.active()) return;

    int64_t i = ti*S.mpi.cart.p + S.mpi.cart.i; // local to global tile index
    int64_t j = tj*S.mpi.cart.q + S.mpi.cart.j;
    if(i >= S.mtile || j >= S.ntile) return; // last row/col

    //Timed timer("Copy Tile", S.inTileMb(i)*S.inTileNb(j)/(1024.*1024*1024), S.mpi);

    int64_t start_m = S.inTileMb(0)*S.mpi.cart.i; // assume const block size
    int64_t start_n = S.inTileNb(0)*S.mpi.cart.j;
    int64_t end_m = start_m + S.inTileMb(i);
    int64_t end_n = start_n + S.inTileNb(j);

    auto St = S(i,j);
    auto U = slice(out, start_m, end_m, start_n, end_n);
    if(verbose) {
        printf("cart = %d,%d and ti,tj = %ld,%ld\n", S.mpi.cart.i, S.mpi.cart.j, ti, tj);
        printf("St(%ld,%ld) is %lu,%lu size %ld %ld %ld\n", i,j,
                (St.data()-S(0,0).data())%St.t->stride,
                (St.data()-S(0,0).data())/St.t->stride,
                St.t->m, St.t->n, St.t->stride);
        printf("U(%ld:%ld,%ld:%ld) is %lu,%lu size %ld %ld %ld\n", start_m,end_m, start_n,end_n,
                (U->data-out->data)%out->stride,
                (U->data-out->data)/out->stride,
                U->m, U->n, U->stride);
    }
    S.mpi.ctxt->copy(St.t, U);
}

template <typename T>
int test(int64_t m, int64_t n, int64_t mbs, int64_t nbs, CartComm &cX, CartComm &cS, const Place loc) {
    auto X = newWaveFn<T>(m, n, mbs, nbs*cS.cart.p, cX, loc);
    auto A = X.alloc(mbs);
    auto S = newMatrix<T>(n, n, nbs, nbs, cS, loc);
    auto B = S.alloc(nbs);

    auto o1 = std::make_shared<Tile<T> >(nbs*S.mpi.cart.p, nbs*S.mpi.cart.q, 1, loc);
    auto o2 = std::make_shared<Tile<T> >(nbs*S.mpi.cart.p, nbs*S.mpi.cart.q, 1, loc);
    if(progress)
        printf("Rank %d: out = %ld %ld @ %p\n", cX.rank, o1->m, o1->n, o1->data);

    const int ntest = 5;
    std::vector<double> results(ntest);
    cX.ctxt->set<T>(A, 1.0);
    cX.ctxt->sync();

    if(X.mpi.rank == 0)
        printf("S.N = %ld, nbs=%ld, ntile=%ld -- %ld^2 steps on %d^2 grid\n",
                S.N, nbs, S.ntile,
                X.ntile, S.mpi.cart.q);

    for(int i=0; i<ntest; i++) {
        Event a[2], b[2]; // Events, named after recording process.
        double time = omp_get_wtime();
        int64_t np = roundup(S.mtile,S.mpi.cart.p)/S.mpi.cart.p;
        int64_t nq = roundup(S.ntile,S.mpi.cart.q)/S.mpi.cart.q;

        #pragma omp parallel num_threads(2)
        {
            int thread = omp_get_thread_num();
            int threads = omp_get_num_threads();
            std::shared_ptr<Tile<T> > out[2] = {o1, o2};
            int64_t k = 0;
            #ifdef ENABLE_CUDA
            cudaStream_t stream = cX.ctxt->get_queue().stream();
            #else
            cudaStream_t stream = nullptr;
            #endif

            // compute the lower triangular part of S
            for(int64_t tj=0; tj < nq; tj++) {
                for(int64_t ti=tj; ti < np; ti++,k++) {
                    if(threads != 2 || thread == 0) {
                        if(k > 1) b[k%2].wait(stream);

                        if(progress)
                            printf("Rank %d: compute tile %ld %ld\n", X.mpi.rank, ti, tj);
                        compute_tile(X, out[k%2], ti, tj);
                        a[k%2].record(stream);
                    }
                    if(threads != 2 || thread == 1) {
                        a[k%2].wait(stream);
                        if(progress)
                            printf("Rank %d: reduce/copy tile %ld %ld\n", X.mpi.rank, ti, tj);
                        Timed timer("Allreduce+Copy Tile", o1->m*o1->n/(1024.*1024*1024), X.mpi);
                        X.mpi.allreduce_sum(out[k%2], out[k%2]);
                        set_local(S, out[k%2], ti, tj);
                        b[k%2].record(stream);
                    }
                    // optional: out.conjTrans and set_local(S,out,tj,ti) [ ti != tj ]
                }
            }
        }
        if(progress)
            printf("Rank %d: sync\n", X.mpi.rank);
        cX.ctxt->sync();
        time = omp_get_wtime() - time;
        results[i] = time;
        if(cX.rank == 0)
            printf("S-time = %f sec., GFLOPS = %f\n", time, 2*(n/1024.)*(n/1024.)*(m/1024.) / time);
    }
    /*if(c.rank == 0)
        print_times(results, sizeof(T)*m*n / (1024.*1024));
    */
    return 0;
}

// find min(floor(sqrt(ranks)), max)
int mat_dim(int ranks, int max) {
    int ans = floor(sqrt((double)ranks));
    return ans < max ? ans : max;
}

int main(int argc, char *argv[]) {
    //int64_t m = 100000; // 4 GB of doubles
    //int64_t n =   5000;
    //int64_t m = 10000; // 40 MB of doubles
    int64_t m = 1000; // 40 MB of doubles
    int64_t n = 500;
    //int64_t mbs = 2048;
    int64_t mbs = 256;
    int64_t nbs = 128;

    if(argc == 5) {
        mbs = atol(argv[1]);
        nbs = atol(argv[2]);
        m = atol(argv[3]) * mbs;
        n = atol(argv[4]) * nbs;
    }

    MPIH      mpi(&argc, &argv);
    // max p limits to 1 tile per rank
    int p = mat_dim(mpi.ranks, roundup(n,nbs)/nbs);
    int q = p;

    auto ctxt = std::make_shared<Context>();
    CartComm cX(CartGroup(mpi.comm, mpi.ranks, 1), ctxt);
    CartComm cS(CartGroup(mpi.comm, p, q), ctxt);
    if(mpi.rank == 0)
        printf("Using %d ranks, p=%d\n", cX.ranks, cS.cart.p);

    int ret = 0;

    if(nbs < 1024 || mbs < 1024) {
        if(mpi.rank == 0)
            printf("Host\n");
        #define call_test(value_t) ret += test<value_t>(m,n,mbs,nbs,cX,cS,Place::Host)
        instantiate_template(call_test)
    }

    #ifdef ENABLE_CUDA
    if(mpi.rank == 0)
        printf("CUDA\n");
    #define call_test2(value_t) ret += test<value_t>(m,n,mbs,nbs,cX,cS,Place::CUDA)
    if(nbs < 1024 || mbs < 1024) {
        instantiate_template(call_test2)
    } else {
        call_test2(float);
    }
    #endif

    return ret;
}
