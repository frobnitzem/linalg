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

    const Place loc; // location for all home tiles
    const int64_t M, N;
    int64_t mtile, ntile; // #tile row,columns

    const std::function<int64_t(int64_t)> inTileMb;
    const std::function<int64_t(int64_t)> inTileNb;
    const std::function<int(Idx)> home; // (home rank)
    std::vector<int64_t> rows, cols; // index to local rows and cols

    CartCommP comm; // MPI communicator specialized for p,q
    // TODO: demote to cache, try out indexed storage for speed
    std::map<Idx,TileP<value_t> > tiles;
    
    Matrix(const int64_t M, const int64_t N,
           const int64_t col1, const Place loc,
           const std::function<int64_t(int64_t)> inTileMb,
           const std::function<int64_t(int64_t)> inTileNb,
           const std::function<int(Idx)> home,
           CartCommP);

    // Allocate space and insert tiles.
    TileP<value_t> alloc(int64_t align);

    TileView<value_t> operator()(int64_t i, int64_t j) {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles.at(idx(i,i)), op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles.at(idx(i,j)))
                                   : TileView<value_t>(tiles.at(idx(j,i)), op_);
    }
    TileView<value_t> const at(int64_t i, int64_t j) const {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles.at(idx(i,i)), op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles.at(idx(i,j)))
                                   : TileView<value_t>(tiles.at(idx(j,i)), op_);
    }
    TileView<value_t> at(int64_t i, int64_t j) {
        if(uplo_ != blas::Uplo::General && i == j)
            return TileView<value_t>(tiles.at(idx(i,i)), op_, uplo_);
        return op_ == blas::Op::NoTrans ? TileView<value_t>(tiles.at(idx(i,j)))
                                   : TileView<value_t>(tiles.at(idx(j,i)), op_);
    }

  protected:
    blas::Op op_;
    blas::Uplo uplo_;
};
