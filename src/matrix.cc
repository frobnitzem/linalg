#include <stdlib.h>
#include <linalg.hh>

namespace Linalg {

// Root tile holding data.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_, int64_t lda_, Place loc_)
       : loc(loc_), m(m_), n(n_), lda(lda_), own_data(true) {
    const size_t sz = lda*n*sizeof(value_t);
    switch(loc) {
    case Place::Host: {
        data = (value_t *)malloc(sz);
    } break;
    case Place::CUDA: {
        CHECKCUDA(cudaMalloc((void **)&data, sz));
    } break;
    default:
        assert(0);
    }
}

// Root tile holding external data.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_, int64_t lda_, value_t *data_, Place loc_)
       : loc(loc_), m(m_), n(n_), lda(lda_), data(data_), own_data(false) {}

// m,n submatrix starting at index i,j of parent tile.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_,
         std::shared_ptr<Tile<value_t> > parent_, int64_t i, int64_t j)
       : loc(parent_->loc), m(m_), n(n_), lda(parent_->lda),
         data(parent_->data + i+lda*j),
         own_data(false), parent(parent_) {
     //printf("Subtile %ld,%ld,%ld @ %ld,%ld\n", m,n,lda, i,j);
}

template <typename value_t>
Tile<value_t>::~Tile() {
        //printf("Called dtor for Tile %ld %ld %ld (%d)\n", m, n, lda, own_data);
    if(own_data) {
        switch(loc) {
        case Place::Host: {
            free(data);
        } break;
        case Place::CUDA: {
            CHECKCUDA(cudaFree(data));
        } break;
        default:
            assert(0);
        }
    }
}

template <typename value_t>
void Tile<value_t>::print() {
    if(loc != Place::Host) { // TODO
        printf("device tile\n");
        return;
    }
    for(int64_t i=0; i<m; i++) {
        for(int64_t j=0; j<n; j++) {
            printf("%.1f ", data[j*lda+i]);
        }
        printf("\n");
    }
}

template <typename value_t>
void Tile<value_t>::fill(const value_t x) {
    int64_t k=0;
    switch(loc) {
    case Place::CUDA: {
        if(x != (value_t)0.0) { // TODO
            printf("nonzero device fill is not implemented.\n");
            assert(0);
        }
        CHECKCUDA( cudaMemset(data, 0, lda*n*sizeof(value_t)) );
        return;
    } break;
    case Place::Host: { 
        #pragma omp parallel for collapse(2)
        for(int64_t j=0; j<n; j++) {
            for(int64_t i=0; i<m; i++) {
                data[j*lda+i] = x;
            }
        }
    } break;
    default:
        assert(0);
    }
}

template <typename value_t>
Matrix<value_t>::Matrix(const int64_t M_, const int64_t N_,
           const std::function<int64_t(int64_t)> inTileMb_,
           const std::function<int64_t(int64_t)> inTileNb_,
           std::shared_ptr<CartGroup> cart_) : M(M_), N(N_),
                inTileMb(inTileMb_), inTileNb(inTileNb_), cart(cart_) {
    int64_t i=0, j=0, m=0, n=0;
    for(i=0; m<M; i++) {
        int64_t tm = inTileMb(i);
        //printf("i: %ld %ld\n", i, tm);
        assert(tm > 0);
        m += tm;
    }
    assert(m == M);
    mtile = i;
    for(j=0; n<N; j++) {
        int64_t tn = inTileNb(j);
        //printf("j: %ld %ld\n", j, tn);
        assert(tn > 0);
        n += tn;
    }
    assert(n == N);
    ntile = j;
}

// Allocate space and insert tiles for local segment of
// this matrix in a scalapack-distributed format over a p,q proc. grid.
// Returns the root tile.
template <typename value_t>
TileP<value_t> Matrix<value_t>::alloc(Place loc) {
    int64_t mloc=0, nloc=0;
    #pragma omp parallel for reduction(+: mloc)
    for(int64_t i=cart->i; i<mtile; i+=cart->p) mloc += inTileMb(i);
    #pragma omp parallel for reduction(+: nloc)
    for(int64_t j=cart->j; j<ntile; j+=cart->q) nloc += inTileNb(j);

    TileP<value_t> root = std::make_shared<Tile<value_t> >(mloc, nloc, mloc, loc);

    int64_t n=0;
    for(int64_t j=cart->j; j<ntile; j+=cart->q) {
        int64_t tn = inTileNb(j);
        int64_t m = 0;
        for(int64_t i=cart->i; i<mtile; i+=cart->p) {
            int64_t tm = inTileMb(i);
            TileP<value_t> x = std::make_shared<Tile<value_t> >(tm, tn, root, m, n);
            tiles[idx(i,j)] = x;
            x->fill(0.0);
            m += tm;
        }
        n += tn;
    }
    return root;
}

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<std::complex<float>>;
template class Matrix<std::complex<double>>;
template class Tile<float>;
template class Tile<double>;
template class Tile<std::complex<float>>;
template class Tile<std::complex<double>>;
}
