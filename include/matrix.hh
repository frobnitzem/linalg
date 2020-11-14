#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

using Idx = std::pair<int64_t, int64_t>;
// helper function for creating idx
static inline Idx idx(int64_t i, int64_t j) {
    return {i,j};
}


/*  Tiles can be either root tiles or submatrix views.
 *  Root tiles can either 'own' their data pointer or
 *  have one imported.  The data pointer can never be null.
 */
template <typename value_t>
struct Tile {
    using value_type = value_t;
    const Place loc;
    const int64_t m, n, lda;
    value_t *data;
    const bool own_data; // true only for some root tiles
    const std::shared_ptr<Tile<value_t> > parent;

    // Root tile holding Host data.
    Tile(int64_t m, int64_t n, int64_t lda, Place);
    // Root tile holding external data.
    Tile(int64_t m, int64_t n, int64_t lda, value_t *data, Place);
    // m,n submatrix starting at index i,j of parent tile.
    Tile(int64_t m, int64_t n,
         std::shared_ptr<Tile<value_t> > parent, int64_t i, int64_t j);
    ~Tile();
    void print();
    void fill(value_t a);
};

template <typename value_t>
using TileP = std::shared_ptr<Tile<value_t> >;

template <typename value_t>
struct Matrix {
    using value_type = value_t;

    const int64_t M, N;
    const std::function<int64_t(int64_t)> inTileMb;
    const std::function<int64_t(int64_t)> inTileNb;
    std::shared_ptr<CartGroup> cart;
    std::map<Idx,TileP<value_t> > tiles;
    int64_t mtile, ntile;

    Matrix(const int64_t M, const int64_t N,
           const std::function<int64_t(int64_t)> inTileMb,
           const std::function<int64_t(int64_t)> inTileNb,
           std::shared_ptr<CartGroup> cart);

    // Allocate space and insert tiles.
    TileP<value_t> alloc(Place);
};
