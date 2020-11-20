#ifndef LINALG_HH
#define LINALG_HH
#define INSIDE_LINALG_HH

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <functional>
#include <vector>
#include <deque>
#include <map>

#include <mpi.h>
#include <omp.h>
#include <blas.hh>

#include "linalg_config.hh"
#include "linalg_cuda.hh"
#include "linalg_nccl.hh"

namespace Linalg {

enum class Place {
    Host, PHI, CUDA, HIP
};

// This is the scalapack method, local = lda
inline void BlockCyclic(const int64_t N,  // number of rows
                        const int64_t nb, // row blocksize
                        const int np, // number of processors per row
                        const int i, // my row #
                        int64_t *tiles,
                        int64_t *last,
                        int64_t *local) {
        // number of super-tiles (guaranteed > 0)
        *tiles = (N+nb*np-1)/(nb*np);
        // global # rows in last super-tile (guaranteed > 0)
        *last  = N - nb*np*(*tiles-1);
        // # rows stored local to processor i
        *local = nb*(*tiles-1) + (i*nb < *last) *
                                 ((i+1)*nb < *last ? nb : *last-i*nb);
        // Note: if i*nb < *last,
        //    then this processor does not participate in the last super-tile
}

#include "tile.hh"
#include "context.hh"
#include "comm.hh"
#include "matrix.hh"

} // namespace Linalg
#undef INSIDE_LINALG_HH
#endif
