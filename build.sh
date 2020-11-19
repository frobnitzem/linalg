#!/bin/bash

LOCALROOT=$PWD/build/inst
BLASPP=$(spack find --format '{prefix}' blaspp@2020.10.02 % gcc@8.4.0)
CUDA_COMPILER=$(spack find --format '{prefix}' cuda@11.1.0 % gcc@8.4.0)/bin/nvcc

set -e
if [ ! -s build.sh ]; then
    echo "Error: This script must be run from the src dir."
    exit 1
fi

[ -d build ] && rm -fr build
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-8 \
      -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER \
      -DCMAKE_PREFIX_PATH=$BLASPP \
      -DCMAKE_INSTALL_PREFIX=$LOCALROOT \
      -DENABLE_NCCL=OFF \
      ..
# -DDISABLE_CUDA=ON \
# -DENABLE_NCCL=OFF \
make -j4

(cd tests && ctest)
make install
