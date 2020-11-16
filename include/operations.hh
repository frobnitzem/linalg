#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

// Per-thread context is required by CUDA impl.
struct Context {
    std::vector<blas::Queue> queue; // for device streams
    Context();

    inline blas::Queue &get_queue() {
        return queue[omp_get_thread_num()];
    }
    template <typename value_t>
    void gemm(const value_t alpha, const TileP<value_t> A,
              const TileP<value_t> B, const value_t beta, TileP<value_t> C);
};
