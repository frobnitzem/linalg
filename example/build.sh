#!/bin/bash
# build the example program against this library

COMPILER=gcc@10.2.0
CUDA=cuda@11.2.0

SPACK=`which spack`
if [[ x$SPACK == x"" ]]; then
  SPACK=/gpfs/alpine/world-shared/stf006/rogersdd/spack/bin/spack
fi
LOCALROOT="$PWD/../inst"
BLASPP=$($SPACK find --format '{prefix}' blaspp % $COMPILER)
NCCL=$($SPACK find --format '{prefix}' nccl % $COMPILER)
CUDAToolkit_ROOT=$($SPACK find --format '{prefix}' $CUDA % $COMPILER)

set -e
if [ ! -s build.sh ]; then
    echo "Error: This script must be run from the src dir."
    exit 1
fi

[ -d build ] && rm -fr build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_CUDA_COMPILER="$CUDAToolkit_ROOT/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
      -DCMAKE_PREFIX_PATH="$BLASPP;$NCCL" \
      -DCMAKE_INSTALL_PREFIX="$LOCALROOT" \
      ..
make -j4

make install
