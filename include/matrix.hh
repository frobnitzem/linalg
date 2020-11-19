#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

using Idx = std::pair<int64_t, int64_t>;
// helper function for creating idx
static inline Idx idx(int64_t i, int64_t j) {
    return {i,j};
}

template <typename value_t>
class Matrix {
  public:
    using value_type = value_t;

    const int64_t M, N;
    const std::function<int64_t(int64_t)> inTileMb;
    const std::function<int64_t(int64_t)> inTileNb;
    std::shared_ptr<CartGroup> cart; // MPI communicator specialized for p,q
    std::map<Idx,TileP<value_t> > tiles;
    int64_t mtile, ntile; // #tile row,columns

    Matrix(const int64_t M, const int64_t N,
           const std::function<int64_t(int64_t)> inTileMb,
           const std::function<int64_t(int64_t)> inTileNb,
           std::shared_ptr<CartGroup> cart);

    // Allocate space and insert tiles.
    TileP<value_t> alloc(Place);

    /*TileView<value_t> operator()(int64_t i, int64_t j) const {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles[idx(i,i)], op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles[idx(i,j)])
                                   : TileView<value_t>(tiles[idx(j,i)], op_);
    }
    TileView<value_t> const&  at(int64_t i, int64_t j) const {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles[idx(i,i)], op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles[idx(i,j)])
                                   : TileView<value_t>(tiles[idx(j,i)], op_);
    }
    TileView<value_t> at(int64_t i, int64_t j) {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles[idx(i,i)], op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles[idx(i,j)])
                                   : TileView<value_t>(tiles[idx(j,i)], op_);
    }*/

  protected:
    blas::Op op_;
    blas::Uplo uplo_;
};
