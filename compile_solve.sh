#! /bin/sh

mkdir -p build
cd build
nvcc -lcublas -lcusparse -arch=native -Xptxas -O3 -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -DNDEBUG -Xcompiler -march=native -DDISABLE_CUSPARSE_DEPRECATED -I../include -o solve ../src/Solve.cpp ../src/UtilKernels.cu ../src/Utils.cpp