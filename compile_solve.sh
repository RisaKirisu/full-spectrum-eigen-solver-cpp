#!/bin/bash

DOUBLE=""
if [[ $# -ne 0 ]] && [[ "$1" == "double" ]]; then
    DOUBLE="-Xcompiler -DUSE_DOUBLE"
fi

mkdir -p build
cd build
nvcc -lcublas -lcusparse -arch=native -Xptxas -O3 -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -DNDEBUG $DOUBLE -Xcompiler -march=native -DDISABLE_CUSPARSE_DEPRECATED -I../include -o solve ../src/Solve.cpp ../src/UtilKernels.cu ../src/Utils.cpp