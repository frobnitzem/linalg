#!/bin/bash

LOCALROOT=$PWD/build/inst

set -e
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$LOCALROOT -DCMAKE_INSTALL_PREFIX=$LOCALROOT ..
make -j4

(cd tests && ctest -N)
make install
