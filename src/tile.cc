#include <stdlib.h>
#include <linalg.hh>

namespace Linalg {

// Root tile holding data.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_, int64_t align_, Place loc_)
       : loc(loc_), m(m_), n(n_), stride(roundup(m_, align_)), own_data(true) {
    const size_t sz = stride*n*sizeof(value_t);
    switch(loc) {
    case Place::Host: {
        #ifdef ENABLE_CUDA
        data = blas::device_malloc_pinned<value_t>(stride*n);
        #else
        data = (value_t *)malloc(sz);
        #endif
    } break;
    case Place::CUDA: {
        data = blas::device_malloc<value_t>(stride*n);
    } break;
    default: assert(0);
    }
}

// Root tile holding external data.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_, int64_t align_,
                    value_t *data_, Place loc_) : loc(loc_),
    m(m_), n(n_), stride(roundup(m_, align_)), data(data_), own_data(false) {}

// m,n submatrix starting at index i,j of parent tile.
template <typename value_t>
Tile<value_t>::Tile(int64_t m_, int64_t n_,
         std::shared_ptr<Tile<value_t> > parent_, int64_t i, int64_t j)
       : loc(parent_->loc), m(m_), n(n_), stride(parent_->stride),
         data(parent_->data + i+stride*j),
         own_data(false), parent(parent_) {
     //printf("Subtile %ld,%ld,%ld @ %ld,%ld\n", m,n,stride, i,j);
}

// Convenient wraper for above [i1,i2) [j1,j2)
template <typename value_t>
TileP<value_t> slice(TileP<value_t> A, int64_t i1, int64_t i2, int64_t j1, int64_t j2) {
    assert(0 <= i1 && i2 <= A->m);
    assert(0 <= j1 && j2 <= A->n);
    assert(i1 <= i2 && j1 <= j2);
    return std::make_shared<Tile<value_t>>(i2-i1, j2-j1, A, i1, i2);
}

template <typename value_t>
Tile<value_t>::~Tile() {
        //printf("Called dtor for Tile %ld %ld %ld (%d)\n", m, n, stride, own_data);
    if(!own_data) {
        return;
    }

    switch(loc) {
    case Place::Host: {
        #ifdef ENABLE_CUDA
        blas::device_free_pinned((void *)data);
        #else
        free(data);
        #endif
    } break;
    case Place::CUDA: {
        blas::device_free((void *)data);
    } break;
    default: assert(0);
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
            printf("%.1f ", data[j*stride+i]);
        }
        printf("\n");
    }
}
#define inst_Tile(value_t) template class Tile<value_t>
instantiate_template(inst_Tile)

}
