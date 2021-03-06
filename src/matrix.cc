#include <stdlib.h>
#include <linalg.hh>

namespace Linalg {

template <typename value_t>
Matrix<value_t>::Matrix(const int64_t M_, const int64_t N_,
           const int64_t col1, const Place loc_,
           const std::function<int64_t(int64_t)> inTileMb_,
           const std::function<int64_t(int64_t)> inTileNb_,
           const std::function<int(Idx)> home_,
           CartComm &mpi_) : loc(loc_), M(M_), N(N_),
                inTileMb(inTileMb_), inTileNb(inTileNb_),
                home(home_), mpi(mpi_),
                op_(blas::Op::NoTrans), uplo_(blas::Uplo::General) {
    int rank = mpi.rank;
    int64_t i=0, j=0, m=0, n=0;
    for(i=0; m<M; i++) {
        int64_t tm = inTileMb(i);
        if(home(Idx(i,col1)) == rank) {
            rows.push_back(i);
        }
        //printf("i: %ld %ld\n", i, tm);
        assert(tm > 0);
        m += tm;
    }
    int64_t row1 = rows.size() > 0 ? rows[0] : -1;

    assert(m == M);
    mtile = i;
    for(j=0; n<N; j++) {
        int64_t tn = inTileNb(j);
        //printf("j: %ld %ld\n", j, tn);
        if(home(Idx(row1,j)) == rank) {
            cols.push_back(j);
        }
        assert(tn > 0);
        n += tn;
    }
    assert(n == N);
    ntile = j;
}

template <typename value_t>
TileP<value_t> Matrix<value_t>::alloc(int64_t align) {
    int64_t mloc=0, nloc=0;
    for(auto i : rows) mloc += inTileMb(i);
    for(auto j : cols) nloc += inTileNb(j);

    TileP<value_t> root = std::make_shared<Tile<value_t> >(mloc, nloc, align, loc);

    int64_t n=0; // running column offset of sub-tile upper-left corner
    for(auto j : cols) {
        int64_t tn = inTileNb(j);
        int64_t m = 0; // running row offset of sub-tile upper-left corner
        for(auto i : rows) {
            int64_t tm = inTileMb(i);
            TileP<value_t> x = std::make_shared<Tile<value_t> >(tm, tn, root, m, n);
            tiles[idx(i,j)] = x;
            m += tm;
        }
        n += tn;
    }
    return root;
}

#define inst_Matrix(value_t) template class Matrix<value_t>
instantiate_template(inst_Matrix)
//#define inst_Tile(value_t) template class Tile<value_t>
//instantiate_template(inst_Tile)
}
