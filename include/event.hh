#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif

#ifndef ENABLE_CUDA
typedef void* cudaEvent_t;
typedef void* cudaStream_t;
#endif

struct Event {
    Event() {
        CHECK0CUDA( cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) );
        sync.lock();
    }
    Event(const Event &) = delete;
    Event &operator=(Event &) = delete;

    ~Event() {
        CHECK0CUDA( cudaEventDestroy(ev) );
    }
    void record(cudaStream_t stream) {
        CHECK0CUDA( cudaEventRecord(ev, stream) );
        sync.unlock();
    }
    void wait(cudaStream_t stream) {
        sync.lock(); // will not succeed until record call has completed
        CHECK0CUDA( cudaEventSynchronize(ev) );
        //CHECK0CUDA( cudaStreamWaitEvent(stream, ev, 0) );
        // leave in the locked state so that another record() is needed.
    }

    std::mutex sync;
    cudaEvent_t ev;
};
