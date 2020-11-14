#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

// Per-thread context is required by CUDA impl.
struct Context {
    std::vector<CUBLAS_HANDLE_T> cublas_handle;
    std::vector<CUDA_STREAM_T>   cuda_stream;
    Context();
    ~Context();

    inline CUBLAS_HANDLE_T get_cublas_handle() {
        return cublas_handle[omp_get_thread_num()];
    }
    template <typename value_t>
    void gemm(const value_t alpha, const TileP<value_t> A,
              const TileP<value_t> B, const value_t beta, TileP<value_t> C);
};
