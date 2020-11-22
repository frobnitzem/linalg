#include <linalg.hh>

namespace Linalg {

template <typename value_t> MPI_Datatype MPI_T();
template <> MPI_Datatype MPI_T<float>()  { return MPI_FLOAT; };
template <> MPI_Datatype MPI_T<double>() { return MPI_DOUBLE; };

#ifdef ENABLE_NCCL
template <typename value_t> ncclDataType_t NCCL_T();
template <> ncclDataType_t NCCL_T<float>()  { return ncclFloat; }
template <> ncclDataType_t NCCL_T<double>() { return ncclDouble; }
#endif

// In-place sum-reduce a tile.
template <typename value_t>
void Comm::allreduce_sum(TileP<value_t> dst, const TileP<value_t> src) {
    blas_error_if_msg(dst->m != src->m || dst->n != src->n || dst->stride != src->stride,
                      "Tile dimensions must match for allreduce.");
    blas_error_if_msg(dst->loc != src->loc,
                      "Cannot change places during allreduce.");
    using T   = typename is_complex_t<value_t>::value_type;
    int scale = is_complex_t<value_t>() ? 2 : 1;
    size_t count = dst->stride*dst->n * scale;

    switch(dst->loc) {
    case Place::Host: {
        if(dst->data == src->data) {
            CHECKMPI( MPI_Allreduce(MPI_IN_PLACE, (void *)dst->data,
                          count, MPI_T<T>(), MPI_SUM, mpi->comm)
                    );
        } else {
            CHECKMPI( MPI_Allreduce((void *)src->data, (void *)dst->data,
                          count, MPI_T<T>(), MPI_SUM, mpi->comm)
                    );
        }
    } break;
    case Place::CUDA: {
        CHECKNCCL(ncclAllReduce(
                  src->data, dst->data,
                  count,
                  NCCL_T<T>(), ncclSum,
                  comm, ctxt->get_queue().stream()));
    } break;
    default: assert(0);
    }
}
#define inst_allreduce_sum(value_t) template void \
        Comm::allreduce_sum(TileP<value_t> dst, const TileP<value_t> src)
instantiate_template(inst_allreduce_sum)

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
