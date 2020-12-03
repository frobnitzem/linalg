#include <linalg.hh>
#include <math.h>
#include "testing.hh"

using namespace Linalg;

static const int verbose = 0;
static const int progress = 0;

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

template <typename T>
int check_matrix_distn(Matrix<T> &M) {
    if(!M.mpi.cart.active() || M.rows.size() == 0 || M.cols.size() == 0) return 0;

    int64_t err1=0, err2=0;
    TileView<T> T0 = M(M.rows[0], M.cols[0]);
    int64_t mb = T0.mb(); // blk-size from first tile
    int64_t nb = T0.nb();
    int64_t stride = T0.t->stride; // entire tile stride
    T *first = T0.data();

    for(int64_t tj : M.cols) {
        for(int64_t ti : M.rows) {
            if(M.home(Idx(ti,tj)) != M.mpi.rank) {
                err1 += 1;
                continue;
            }
            T *tile = M(ti,tj).data();
            int64_t i = ( (tile-first)%stride ) / mb;
            int64_t j = ( (tile-first)/stride ) / nb;
            err2 += (M.rows[i] != ti) || (M.cols[j] != tj);
        }
    }
    if(err1 > 0)
        printf("Rank %d: missing %ld home tiles\n", M.mpi.rank, err1);
    if(err2 > 0)
        printf("Rank %d: improper location for %ld tiles\n", M.mpi.rank, err2);

    return err1 > 0 || err2 > 0;
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
    if(! S.mpi.cart.active()) return;

    int64_t i = ti*S.mpi.cart.p + S.mpi.cart.i; // local to global tile index
    int64_t j = tj*S.mpi.cart.q + S.mpi.cart.j;
    if(i >= S.mtile || j >= S.ntile) return; // last row/col

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
    if( check_matrix_distn<T>(X) ) return 1;
    if( check_matrix_distn<T>(S) ) return 1;

    auto out = std::make_shared<Tile<T> >(nbs*S.mpi.cart.p, nbs*S.mpi.cart.q, 1, loc);
    if(progress)
        printf("Rank %d: out = %ld %ld @ %p\n", cX.rank, out->m, out->n, out->data);

    const int ntest = 5;
    std::vector<double> results(ntest);
    cX.ctxt->set<T>(A, 1.0);
    cX.ctxt->sync();

    if(X.mpi.rank == 0)
        printf("S.N = %ld, nbs=%ld, ntile=%ld -- %ld^2 steps on %d^2 grid\n",
                S.N, nbs, S.ntile,
                X.ntile, S.mpi.cart.q);

    for(int i=0; i<ntest; i++) {
        double time = omp_get_wtime();
        // compute the lower triangular part of S
        for(int64_t tj=0; tj < roundup(S.ntile,S.mpi.cart.q)/S.mpi.cart.q; tj++) {
            for(int64_t ti=tj; ti < roundup(S.mtile,S.mpi.cart.p)/S.mpi.cart.p; ti++) {
                if(progress)
                    printf("Rank %d: compute tile %ld %ld\n", X.mpi.rank, ti, tj);
                compute_tile(X, out, ti, tj);
                if(progress)
                    printf("Rank %d: reduce tile %ld %ld (%d)\n", X.mpi.rank, ti, tj, out->data == out->data);
                X.mpi.allreduce_sum(out, out);
                if(progress)
                    printf("Rank %d: copy tile %ld %ld\n", X.mpi.rank, ti, tj);
                set_local(S, out, ti, tj);
                // optional: out.conjTrans and set_local(S,out,tj,ti) [ ti != tj ]
            }
        }
        if(progress)
            printf("Rank %d: sync\n", X.mpi.rank);
        cX.ctxt->sync();
        time = omp_get_wtime() - time;
        results[i] = time;
        if(cX.rank == 0)
            printf("S-time = %f sec., GFLOPS = %f\n", time, (n/1024.)*(n/1024.)*(m/1024.));
    }
    if(cX.rank == 0)
        print_times(results, 2*(n/1024.)*(n/1024.)*(m/1024.));

    if(! S.mpi.cart.active()) return 0;

    auto C = std::make_shared<Tile<T> >(B->m, B->n, B->stride, Place::Host);
    cX.ctxt->set<T>(C, m);
    TileP<T> Bx = B;
    if(loc == Place::CUDA) {
        Bx = std::make_shared<Tile<T> >(B->m, B->n, B->stride, Place::Host);
        cX.ctxt->copy(Bx, B);
    }
    cX.ctxt->sync();
    double err = 0.0; // only check the lower-diagonal
    for(auto j : S.cols) {
        int64_t off = ((j-cS.cart.j)/cS.cart.q)*nbs; // local offset of col j in root tile
        err += nrm( slice(Bx, off, Bx->m, off, off+S.inTileNb(j)),
                    slice(C,  off, C->m,  off, off+S.inTileNb(j)) );
    }
    printf("ans = %f, expected = %f, err = %f\n", std::abs(Bx->at(0,0)), std::abs(C->at(0,0)), err);
    return err > 1e-8;
}

// find min(floor(sqrt(ranks)), max)
int mat_dim(int ranks, int max) {
    int ans = floor(sqrt((double)ranks));
    return ans < max ? ans : max;
}

int main(int argc, char *argv[]) {
    //int64_t m = 100000; // 4 GB of doubles
    //int64_t n = 5000;
    //int64_t m = 10000; // 40 MB of doubles
    int64_t m = 1000; // 0.4 MB of doubles
    int64_t n = 500;
    //int64_t mbs = 2048;
    int64_t mbs = 256;
    int64_t nbs = 128;

    if(argc == 5) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        mbs = atol(argv[3]);
        nbs = atol(argv[4]);
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

    if(mpi.rank == 0)
        printf("Host\n");
#define call_test(value_t) ret += test<value_t>(m,n,mbs,nbs,cX,cS,Place::Host)
    instantiate_template(call_test)

    #ifdef ENABLE_CUDA
    if(mpi.rank == 0)
        printf("CUDA\n");
#define call_test2(value_t) ret += test<value_t>(m,n,mbs,nbs,cX,cS,Place::CUDA)
    instantiate_template(call_test2)
    #endif

    return ret;
}
