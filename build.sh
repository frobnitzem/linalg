#!/bin/bash

#module purge
#module load gcc/8.1.1 cmake/3.18 spectrum-mpi cuda/11.1.1
#module load gcc/6.4.0 cmake/3.18 spectrum-mpi cuda/10.2.89

COMPILER=$(module list gcc 2>&1 | sed -n -e 's/.*gcc\//gcc@/p')
CUDA=$(module list cuda 2>&1 | sed -n -e 's/.*cuda\//cuda@/p')
CUDA_ARCH=70
if [[ x$COMPILER == x"" ]]; then
    COMPILER=gcc@8.4.0
    CUDA=cuda@11.1.0
    CUDA_ARCH=75
fi

# build blaspp with $SPACK install -v --no-cache -j4 blaspp@2020.10.02 % $COMPILER +cuda cuda_arch=$CUDA_ARCH ^$CUDA ^openblas+ilp64 threads=openmp
# build nccl with $SPACK install -v -j4 --no-cache nccl % $COMPILER +cuda cuda_arch=$CUDA_ARCH ^$CUDA

SPACK=`which spack`
if [[ x$SPACK == x"" ]]; then
  SPACK=/gpfs/alpine/proj-shared/eng110/spack/bin/spack
fi
LOCALROOT=$PWD/build/inst
BLASPP=$($SPACK find --format '{prefix}' blaspp@2020.10.02 % $COMPILER)
NCCL=$($SPACK find --format '{prefix}' nccl % $COMPILER)
CUDAToolkit_ROOT=$($SPACK find --format '{prefix}' $CUDA % $COMPILER)

set -e
if [ ! -s build.sh ]; then
    echo "Error: This script must be run from the src dir."
    exit 1
fi

disable_cuda=OFF
while [ $# -ge 1 ]; do
    opt="$1"
    shift;
    case "$opt" in
        nocuda)
            disable_cuda=ON
            ;;
        *)
            echo "Unrecognized option: '$opt'"
            exit 1
            ;;
    esac
done

[ -d build ] && rm -fr build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_CUDA_COMPILER="$CUDAToolkit_ROOT/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
      -DCMAKE_PREFIX_PATH="$BLASPP;$NCCL" \
      -DCUDA_ARCH=$CUDA_ARCH \
      -DCMAKE_INSTALL_PREFIX="$LOCALROOT" \
      -DDISABLE_CUDA=$disable_cuda \
      ..
make -j4

(cd tests && ctest)
make install
