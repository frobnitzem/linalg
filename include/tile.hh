#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

/*  Tiles can be either root tiles or submatrix views.
 *  Root tiles can either 'own' their data pointer or
 *  have one imported.  The data pointer can never be null.
 */
template <typename value_t>
struct Tile {
    using value_type = value_t;
    const Place loc;
    const int64_t m, n, stride;
    value_t *data;
    const bool own_data; // true only for some root tiles
    const std::shared_ptr<Tile<value_t> > parent;

    // Root tile managing data pointer.
    Tile(int64_t m, int64_t n, int64_t stride, Place);
    // Root tile holding external data pointer.
    Tile(int64_t m, int64_t n, int64_t stride, value_t *data, Place);
    // m,n submatrix starting at index i,j of parent tile
    //     (points to external data)
    Tile(int64_t m, int64_t n,
         std::shared_ptr<Tile<value_t> > parent, int64_t i, int64_t j);
    ~Tile();
    value_t operator()(int64_t i, int64_t j) const { return data[j*stride+i]; }
    value_t const&  at(int64_t i, int64_t j) const { return data[j*stride+i]; }
    value_t&        at(int64_t i, int64_t j)       { return data[j*stride+i]; }
    void print();
};
template <typename value_t>
using TileP = std::shared_ptr<Tile<value_t> >;

/* The TileView encapsulates a tile with transposition and symmetry annotations */
template <typename value_t>
class TileView {
  public:
    using value_type = value_t;
    TileP<value_t> t;

    TileView(TileP<value_t> _t,
             blas::Op _op = blas::Op::NoTrans,
             blas::Uplo _uplo = blas::Uplo::General) :
        t(_t), op_(_op), uplo_(_uplo) {}

    // rows
    int64_t mb() const { return (op_ == blas::Op::NoTrans ? t->m : t->n); }
    // cols
    int64_t nb() const { return (op_ == blas::Op::NoTrans ? t->n : t->m); }
    // lda
    int64_t stride() const { return t->stride; }

    /// Returns const pointer to data, i.e., A(0,0), where A is this tile
    value_t const* data() const { return t->data; }

    // Returns pointer to data, i.e., A(0,0), where A is this tile
    value_t*       data()       { return t->data; }

    value_t operator()(int64_t i, int64_t j) const {
        assert(op_ != blas::Op::ConjTrans);
        return op_ == blas::Op::NoTrans ? t->at(i,j) : t->at(j,i);
    }
    value_t const&  at(int64_t i, int64_t j) const {
        debug_assert(op_ != blas::Op::ConjTrans);
        return op_ == blas::Op::NoTrans ? t->at(i,j) : t->at(j,i);
    }
    value_t&        at(int64_t i, int64_t j)       {
        debug_assert(op_ != blas::Op::ConjTrans);
        return op_ == blas::Op::NoTrans ? t->at(i,j) : t->at(j,i);
    }

    void trans() { // transpose in-place
        switch(op_) {
        case blas::Op::NoTrans: op_ = blas::Op::Trans; break;
        case blas::Op::Trans: op_ = blas::Op::NoTrans; break;
        default:
            assert(0); // not supported
        }
    }
    void conjTrans()      { // conj-transpose in-place
        switch(op_) {
        case blas::Op::NoTrans: op_ = blas::Op::ConjTrans; break;
        case blas::Op::ConjTrans: op_ = blas::Op::NoTrans; break;
        default:
            assert(0); // not supported
        }
    }

    blas::Op op() const     { return op_; }   // operator: NoTrans|Trans|ConjTrans
    blas::Uplo uplo() const { return uplo_; } // storage: Upper|Lower|General

    /// Returns which host or GPU device tile's data is located on.
    Place device() const { return t->loc; }

  protected:
    blas::Op op_;
    blas::Uplo uplo_;
};