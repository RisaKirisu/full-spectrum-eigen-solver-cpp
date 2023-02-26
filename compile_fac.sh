#! /bin/sh

mkdir -p build
cd build
g++ -std=c++17 -O3 -fopenmp -DNDEBUG -march=native -I../include -o factorize ../src/Factorize.cpp ../src/Utils.cpp