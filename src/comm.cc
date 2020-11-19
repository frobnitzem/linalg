#include <linalg.hh>

namespace Linalg {

template <typename value_t> struct NCCL_T {};
#ifdef ENABLE_NCCL
template <> struct NCCL_T<float> { static constexpr ncclDataType_t type = ncclFloat; };
template <> struct NCCL_T<double> { static constexpr ncclDataType_t type = ncclDouble;  };
#else
template <> struct NCCL_T<float> { static constexpr MPI_Datatype type = MPI_FLOAT; };
template <> struct NCCL_T<double> { static constexpr MPI_Datatype type = MPI_DOUBLE; };
#endif

// Template magic to compute types and sizes.
template <typename T> struct CommValue {
    using type = typename NCCL_T< typename is_complex_t<T>::value_type >::type;
    static constexpr int scale = is_complex_t<T>() ? 2 : 1;
};

// In-place sum-reduce a tile.
// FIXME: make this part of the context (so cuda knows its stream)
template <typename value_t>
void NCCL::reduce_sum(TileP<value_t> tile) {
    using comm_val_t = CommValue<value_t>;
    size_t count = tile->stride*tile->n * comm_val_t::scale;

    switch(tile->loc) {
    case Place::Host: {
        CHECKMPI( MPI_Allreduce(MPI_IN_PLACE, (void *)tile->data,
                      count, comm_val_t::type, MPI_SUM, mpi->comm)
                );
    } break;
    case Place::CUDA: {
        CHECKNCCL(ncclAllReduce(
                  tile->data, tile->data,
                  count,
                  comm_val_t::type, ncclSum,
                  comm, stream));
    }
    default: assert(0);
    }
}

}
/*
int main(int argc, char *argv[]) {
    auto mpi = std::make_shared<MPIH>(&argc, &argv);
    auto nccl = std::make_shared<NCCLH>(mpi);

    float *sendbuff, *recvbuff;
    CHECKCUDA(cudaMalloc(&sendbuff, size * sizeof(float)));
    CHECKCUDA(cudaMalloc(&recvbuff, size * sizeof(float)));

    std::cout << "Hello" << std::endl;
    run(nccl, sendbuff, recvbuff);
    CHECKCUDA(cudaStreamSynchronize(nccl->stream));

    return 0;
}*/
