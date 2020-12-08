#!/bin/bash
# build the example program against this library

#module purge
#module load gcc/8.1.1 cmake/3.18 spectrum-mpi cuda/11.1.1
#module load gcc/6.4.0 cmake/3.18 spectrum-mpi cuda/10.2.89

COMPILER=$(module list gcc 2>&1 | sed -n -e 's/.*gcc\//gcc@/p')
CUDA=$(module list cuda 2>&1 | sed -n -e 's/.*cuda\//cuda@/p')
if [[ x$COMPILER == x"" ]]; then
    COMPILER=gcc@8.4.0
    CUDA=cuda@11.1.0
fi

SPACK=`which spack`
if [[ x$SPACK == x"" ]]; then
  SPACK=/gpfs/alpine/proj-shared/eng110/spack/bin/spack
fi
LOCALROOT="$PWD/../build/inst"
BLASPP=$($SPACK find --format '{prefix}' blaspp@2020.10.02 % $COMPILER)
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
