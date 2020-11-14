#!/bin/bash

LOCALROOT=$PWD/build/inst
#OPENBLAS=/home/99r/spack/opt/spack/linux-ubuntu18.04-skylake/gcc-7.5.0/openblas-0.3.10-yuemfaycuacsveabwu2xflae74ja5xtd
OPENBLAS=/home/99r/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-6.5.0/openblas-0.3.5-ymh7xvk33wxvaqbiv5jjlkyzixldp45y

set -e
if [ ! -s build.sh ]; then
    echo "Error: This script must be run from the src dir."
    exit 1
fi

[ -d build ] && rm -fr build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$OPENBLAS -DCMAKE_INSTALL_PREFIX=$LOCALROOT ..
make -j4

(cd tests && ctest -N)
make install
