#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

// Per-thread context is required by CUDA impl.
struct Context {
    std::deque<blas::Queue> queue; // for device streams
    // TODO: change to std::vector once upstream functionality is present
    Context();

    inline blas::Queue &get_queue(Place loc) {
        assert(loc >= 0 && loc < queue.size());
        return queue[loc];
    }
    template <typename value_t>
    void gemm(const value_t alpha, const TileP<value_t> A,
              const TileP<value_t> B, const value_t beta, TileP<value_t> C);
};
