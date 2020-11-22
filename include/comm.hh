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

struct MPIH {
    int ranks, rank;
    MPI_Comm comm;

    MPIH(int *argc, char **argv[]) : comm(MPI_COMM_WORLD) {
        int provided;
        CHECKMPI( MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) );
        assert(provided >= MPI_THREAD_FUNNELED);
        CHECKMPI( MPI_Comm_size( comm, &ranks) );
        CHECKMPI( MPI_Comm_rank( comm, &rank ) );
    }
    ~MPIH() {
        MPI_Finalize();
    }
};
using MPIp = std::shared_ptr<MPIH>;

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

struct Comm {
    MPIp mpi;
    ContextP ctxt;

    #ifndef ENABLE_NCCL
    Comm(MPIp _mpi, ContextP _ctxt) : mpi(_mpi), ctxt(_ctxt) {}
    #else
    ncclUniqueId id;
    ncclComm_t comm;

    Comm(MPIp _mpi, ContextP _ctxt) : mpi(_mpi), ctxt(_ctxt) {
        if(mpi->rank == 0) CHECKNCCL( ncclGetUniqueId(&id) );
        CHECKMPI( MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, mpi->comm) );
        CHECKNCCL( ncclCommInitRank(&comm, mpi->ranks, id, mpi->rank) );
    }
    ~Comm() {
        CHECKNCCL( ncclCommDestroy(comm) );
    }
    #endif

    Comm (Comm &) = delete;
    Comm &operator=(Comm &) = delete;

    // In-place sum-reduce a tile.
    template <typename value_t>
        void allreduce_sum(TileP<value_t> dst, const TileP<value_t> src);
};
using CommP = std::shared_ptr<Comm>;
