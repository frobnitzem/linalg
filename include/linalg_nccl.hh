#ifndef INSIDE_LINALG_HH
#error "This file should not be included directly.  It is part of linalg.hh"
#endif

#ifndef ENABLE_NCCL
// stubs
    #define CHECKNCCL(cmd) assert(0);

#else // ENABLE_NCCL

#include <nccl.h>

#define CHECKNCCL(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#endif
