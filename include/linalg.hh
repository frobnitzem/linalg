#ifndef LINALG_HH
#define LINALG_HH
#define INSIDE_LINALG_HH

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <memory>
#include <functional>
#include <vector>
#include <map>
#include <complex>

#include <mpi.h>
#include <omp.h>
#include <blas.hh>

#include "linalg_config.hh"
#include "linalg_cuda.hh"

namespace Linalg {

enum class Place {
    Host, PHI, CUDA, HIP
};

// Create a Subgroup with 2D Cartesian Topology
struct CartGroup {
    const int p, q;
    MPI_Comm comm; // == MPI_COMM_NULL for non-participating ranks
    int rank;
    union {
        struct {
            int i, j;
        };
        int coords[2];
    };

    CartGroup(MPI_Comm parent, int p_, int q_) : p(p_), q(q_) {
        MPI_Comm subcomm; // == MPI_COMM_NULL for non-participating ranks
        // Get the group of processes in parent
        MPI_Group group, parent_group;
        assert( !MPI_Comm_group(parent, &parent_group) );
        int size;
        assert( !MPI_Comm_size(parent, &size) );
        assert( size >= p*q ); // Need to have enough ranks to make the decomposition.

        std::vector<int> ranks(p*q);
        for(int k=0; k<p*q; k++) {
            ranks[k] = k;
        }

        // Construct a group containing all of the ranks in parent_group
        assert( !MPI_Group_incl(parent_group, p*q, &ranks[0], &group) );

        // Create a new communicator based on the group
        assert( !MPI_Comm_create_group(parent, group, 0, &subcomm) );

        // Using MPI_COMM_NULL for MPI_Comm_rank or MPI_Comm_size is erroneous.
        if (subcomm != MPI_COMM_NULL) {
            assert( ! MPI_Comm_size(subcomm, &size) );
            assert(size == p*q);
        }

        assert( ! MPI_Group_free(&parent_group) );
        assert( ! MPI_Group_free(&group) );

        if (subcomm != MPI_COMM_NULL) {
            int dims[2] = {p, q};
            int periods[2] = {0, 0};
            assert( ! MPI_Cart_create(subcomm, 2, dims, periods, 1, &comm) );
            assert( ! MPI_Comm_rank(  comm, &rank) );
            assert( ! MPI_Cart_coords(comm, rank, 2, coords) );

            MPI_Comm_free(&subcomm);
        } else {
            comm = MPI_COMM_NULL;
        }
    }

    ~CartGroup() {
        MPI_Comm_free(&comm);
    }
};
using CartGroupP = std::shared_ptr<CartGroup>;

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

#include "matrix.hh"
#include "operations.hh"

} // namespace Linalg
#undef INSIDE_LINALG_HH
#endif
