#include <linalg.hh>

namespace Linalg {

// TODO: Warn the user if there are unused devices.
// - we assume 1 device per MPI rank -
Context::Context() {
    #ifdef ENABLE_CUDA
    //int rank;
    //CHECKMPI( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int rank = 0;
    if(nl_rank != NULL) rank = atoi(nl_rank);
    int devices;
    CHECKCUDA( cudaGetDeviceCount(&devices) );
    int device = rank % devices;
    blas::set_device(device);
    int threads = omp_get_max_threads();
    for(int i=0; i<threads; i++) {
        queue.emplace_back((blas::Device)device, 0);
    }
    #endif
}

}
