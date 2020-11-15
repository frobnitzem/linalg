#!/bin/bash

LOCALROOT=$PWD/build/inst
BLASPP=/home/99r/spack/opt/spack/linux-ubuntu18.04-skylake/gcc-8.4.0/blaspp-2020.10.02-sfs5es6iyemz3y4ynbbfn7z4ea4rthlh
CUDA_COMPILER=/home/99r/spack/opt/spack/linux-ubuntu18.04-skylake/gcc-8.4.0/cuda-11.1.0-jwbmvvabp4ldton37gwq7v577ldkxkfw/bin/nvcc

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
      -DCMAKE_INSTALL_PREFIX=$LOCALROOT ..
make -j4

(cd tests && ctest -N)
make install
