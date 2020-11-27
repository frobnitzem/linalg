#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

///< Index to tiles
using Idx = std::pair<int64_t, int64_t>;

///< Helper function for creating Idx
static inline Idx idx(int64_t i, int64_t j) {
    return {i,j};
}

/**
 * The Matrix class is designed to be a container holding local
 * tiles.  It may eventually cache remote tiles for fast recall.
 *
 * Matrix layout is defined through inTileMb (and inTileNb),
 * which return the rows (columns) in each tile row (column).
 *
 * Matrix distribution over ranks is defined by CartComm mpi,
 * containing a logical grid of p,q processors, as well
 * as home, a function returning the rank owning a given tile.
 * 
 * The local tile-set is determined through calling `home`
 * N+M times during Matrix init.  This assumes a regular
 * distribution, where {ti,tj} and {ti',tj'} belong to this rank
 * as long as {ti',tj} and {ti,tj'} do.
 *
 * In other words, local tiles are all {ti,tj} where ti
 * is in the row-set and tj is in the column-set.
 *
 * The result of constructing a Matrix without such
 * a distribution is undefined, but probably
 * a tile lookup error.
 *
 */
template <typename value_t>
class Matrix {
  public:
    using value_type = value_t; ///< type of matrix elements

    const Place loc; ///< location for all home tiles
    const int64_t M, N; ///< Overall matrix dimensions
    int64_t mtile, ntile; ///< #tile row,columns

    const std::function<int64_t(int64_t)> inTileMb; ///< rows in tile row-blk i
    const std::function<int64_t(int64_t)> inTileNb; ///< cols in tile col-blk j
    const std::function<int(Idx)> home; ///< home rank of ea. tile
    std::vector<int64_t> rows, cols; ///< index to local rows and cols

    CartComm mpi; ///< MPI communicator specialized for p,q
    // TODO: demote to cache, try out indexed storage for speed
    std::map<Idx,TileP<value_t> > tiles; ///< individual tiles
    
    /**
     * Construct a Matrix without allocating any tile data.
     *
     * @param col1 is the tile index to the first column
     *        which this processor owns.
     */
    Matrix(const int64_t M, const int64_t N,
           const int64_t col1, const Place loc,
           const std::function<int64_t(int64_t)> inTileMb,
           const std::function<int64_t(int64_t)> inTileNb,
           const std::function<int(Idx)> home,
           CartComm &);

    /**
     * Allocate dense storage space and insert tiles.
     * Tile strides are chosen based on inTileMb,
     * rounded up to the given alignment.
     *
     * @return A super-tile holding all local values in the matrix.
     *         All local tiles (e.g. returned by tiles[]) reference
     *         some sub-tile of this super-tile with very large stride.
     *         Its layout is colum-major, compatible with scalapack.
     */
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
