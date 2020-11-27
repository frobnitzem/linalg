#!/bin/bash
# build the example program against this library

LOCALROOT="$PWD/../build/inst"
BLASPP=$(spack find --format '{prefix}' blaspp@2020.10.02 % gcc@8.4.0)
NCCL=$(spack find --format '{prefix}' nccl % gcc@8.4.0)
CUDAToolkit_ROOT=$(spack find --format '{prefix}' cuda@11.1.0 % gcc@8.4.0)

set -e
if [ ! -s build.sh ]; then
    echo "Error: This script must be run from the src dir."
    exit 1
fi

[ -d build ] && rm -fr build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=gcc-8 \
      -DCMAKE_CXX_COMPILER=g++-8 \
      -DCMAKE_CUDA_COMPILER="$CUDAToolkit_ROOT/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
      -DCMAKE_PREFIX_PATH="$BLASPP;$NCCL" \
      -DCMAKE_INSTALL_PREFIX="$LOCALROOT" \
      ..
#     -DDISABLE_CUDA=ON \
make -j4

make install
