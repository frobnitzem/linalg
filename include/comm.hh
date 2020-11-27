#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

#define CHECKMPI(cmd) do {                        \
    int e = cmd;                                  \
    if( e != MPI_SUCCESS ) {                      \
        printf("Failed: MPI error %s:%d '%d'\n",  \
            __FILE__,__LINE__, e);                \
        exit(EXIT_FAILURE);                       \
    }                                             \
} while(0)

/**! Used internally for ref-counting by MPIH. */
using MPIp = std::shared_ptr<MPI_Comm>;

/**
 * MPI wrapper to manage MPI_Init/Finalize (or Comm_free)
 * and provide rank info quickly.  Pass by reference or value, either way.
 * 
 * The constructor from argc, argv should only be called once.
 *
 * Note that the default constructor for this object will create an
 * uninitialized communicator, which you should set on your own.
 * It will be freed for you by calling MPI_Comm_free.
 *
 * This class internally uses a shared pointer to count the number of copies.
 * It works because the compiler-generated copy and assignment rules also
 * copy the shared pointer.
 * This avoids calling MPI_Comm_dup, and has the effect of keeping all
 * communicators "live" until no more MPIH structs use it.
 *
 */
struct MPIH {
    int ranks, rank;
    MPI_Comm comm;

    MPIH(int *argc, char **argv[]) : comm(MPI_COMM_WORLD), pcomm(&comm, MPIH::free_mpi) {
        int provided;
        CHECKMPI( MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) );
        assert(provided >= MPI_THREAD_FUNNELED);
        CHECKMPI( MPI_Comm_size( comm, &ranks) );
        CHECKMPI( MPI_Comm_rank( comm, &rank ) );
    }

    MPIH() : rank(-1), ranks(0), comm(nullptr), pcomm(&comm, MPIH::free_comm) { }

    private:
    MPIp pcomm;

    static void free_mpi(MPI_Comm *p) { MPI_Finalize(); }
    static void free_comm(MPI_Comm *p) { if(*p != nullptr) MPI_Comm_free(p); }
};

/**
 * A Subgroup with 2D Cartesian Topology.
 */
struct CartGroup : MPIH {
    const int p, q;
    union {
        struct {
            int i, j;
        };
        int coords[2];
    };

    /**
     * Create a new subgroup containing ranks start .. start+p*q-1
     */
    CartGroup(MPI_Comm parent, int p_, int q_, int start=0) : p(p_), q(q_) {
        MPI_Comm subcomm; // == MPI_COMM_NULL for non-participating ranks
        // Get the group of processes in parent
        MPI_Group group, parent_group;
        assert( !MPI_Comm_group(parent, &parent_group) );
        int size;
        assert( !MPI_Comm_size(parent, &size) );
        assert( size >= p*q ); // Need to have enough ranks to make the decomposition.

        ranks = p*q;
        std::vector<int> ingrp(ranks);
        for(int k=0; k<ranks; k++) {
            ingrp[k] = (k+start) % size;
        }

        // Construct a group containing all of the ranks in parent_group
        assert( !MPI_Group_incl(parent_group, ranks, &ingrp[0], &group) );

        // Create a new communicator based on the group
        assert( !MPI_Comm_create_group(parent, group, 0, &subcomm) );

        // Using MPI_COMM_NULL for MPI_Comm_rank or MPI_Comm_size is erroneous.
        if (subcomm != MPI_COMM_NULL) {
            assert( ! MPI_Comm_size(subcomm, &size) );
            assert(size == ranks);
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
};

/**
 * Helper class to store NCCL communicator.
 * Like MPIH, it uses a shared ptr to manage the communicator.
 */
struct NCCLH {
    ContextP ctxt;

    #ifndef ENABLE_NCCL
    NCCLH(MPIH &mpi, ContextP _ctxt) : ctxt(_ctxt) {}
    #else
    ncclComm_t ncom;

    NCCLH(MPIH &mpi, ContextP _ctxt) : ctxt(_ctxt), pncom(&ncom, NCCLH::dtor) {
        ncclUniqueId id;
        if(mpi.rank == 0) CHECKNCCL( ncclGetUniqueId(&id) );
        CHECKMPI( MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, mpi.comm) );
        CHECKNCCL( ncclCommInitRank(&ncom, mpi.ranks, id, mpi.rank) );
    }
    #endif

    #ifdef ENABLE_NCCL
    private:
    std::shared_ptr<ncclComm_t> pncom;
    static void dtor(ncclComm_t *p) {
        CHECKNCCL( ncclCommDestroy(*p) );
    }
    #endif
};

struct Comm : MPIH, NCCLH {
    Comm(MPIH &mpi, ContextP ctxt) : MPIH(mpi), NCCLH(mpi, ctxt) {}

    /**
     * Sum a tile over all ranks.  dst and src must be on the same
     * device types and have the same layout.  In-place summation
     * is done if dst and src point to the same data.
     */
    template <typename value_t>
        void allreduce_sum(TileP<value_t> dst, const TileP<value_t> src);
};

/**
 * Trivial extension of Comm holding cart.
 */
struct CartComm : Comm {
    CartGroup cart; ///< Holds a copy of the communicator to avoid inheritance triangle problem.
    CartComm(CartGroup &_cart, ContextP ctxt) : Comm(_cart, ctxt), cart(_cart) {}
};
