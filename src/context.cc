#include <linalg.hh>

namespace Linalg {

Context::Context() {
    #ifdef ENABLE_CUDA
    int rank;
    CHECKMPI( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    // TODO: Warn the user if there are unused devices.
    // - we assume 1 device per MPI rank -
    //int ranks; // (would need #nodes)
    //CHECKMPI( MPI_Comm_size(MPI_COMM_WORLD, &ranks) );
    int devices;
    CHECKCUDA( cudaGetDeviceCount(&devices) );
    //CUDACHECK( cudaSetDevice(rank % devices) );
    blas::set_device(rank % devices);
    //const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK"); // could try?
    //int node_local_rank = atoi(nl_rank);
    int threads = omp_get_max_threads();
    for(int i=0; i<threads; i++) {
        queue.emplace_back((blas::Device)device, 0);
    }
    #endif
}

}
