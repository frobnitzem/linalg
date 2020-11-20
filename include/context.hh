#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif
// Note: this is inside namespace Linalg

// Per-GPU context is required by CUDA impl.
struct Context {
    std::deque<blas::Queue> queue;
    Context();

    inline blas::Queue &get_queue() {
        return queue[omp_get_thread_num()];
    }
    void sync() {
        get_queue().sync();
    }
    template <typename value_t>
    void gemm(const value_t alpha, const TileP<value_t> A,
              const TileP<value_t> B, const value_t beta, TileP<value_t> C);
    template <typename value_t>
    void gemm(const value_t alpha, const TileView<value_t> A,
              const TileView<value_t> B, const value_t beta, TileView<value_t> C);
    template <typename value_t>
    void set(TileP<value_t>, const value_t);
    template <typename value_t>
    void set(TileView<value_t> A, const value_t a) { set(A.t, a); };

    template <typename value_t>
    void copy(TileP<value_t> dst, TileP<value_t>src);
    template <typename dst_t, typename src_t>
    void copy(TileView<dst_t> dst, const TileView<src_t> src);

    #ifdef ENABLE_CUDA
    template <typename value_t>
    void set_cuda(TileP<value_t>, const value_t);
    template <typename dst_t, typename src_t>
    void copy_cuda(TileView<dst_t>, const TileView<src_t>);
    #endif
};
using ContextP = std::shared_ptr<Context>;
