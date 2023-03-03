#! /bin/bash

DOUBLE=""
if [[ $# -ne 0 ]] && [[ "$1" == "double" ]]; then
    DOUBLE="-DUSE_DOUBLE"
fi

mkdir -p build
cd build
g++ -std=c++17 -O3 -fopenmp -DNDEBUG $DOUBLE -march=native -I../include -o factorize ../src/Factorize.cpp ../src/Utils.cpp