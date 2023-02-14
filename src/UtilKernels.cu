#include <cuComplex.h>
#include <algorithm>
#include "CudaHelper.h"
#include <cstdio>

namespace GPU {

template <typename Scalar>
__global__ void permuteKernel(const Scalar* __restrict__ v, const int* __restrict__ perm, Scalar* __restrict__ dst, int size)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    dst[perm[i]] = v[i];
  }
}

template <typename Scalar>
__global__ void addInPlaceKernel(Scalar * __restrict__ lhs, Scalar * __restrict__ rhs)
{
  *lhs += *rhs;
}

template <>
__global__ void addInPlaceKernel<cuComplex>(cuComplex *__restrict__ a, cuComplex *__restrict__ b)
{
  a->x += b->x;
  a->y += b->y;
}

template <>
__global__ void addInPlaceKernel<cuDoubleComplex>(cuDoubleComplex *__restrict__ a, cuDoubleComplex *__restrict__ b)
{
  a->x += b->x;
  a->y += b->y;
}

template <typename Scalar, typename RealType>
__global__ void divideByRealKernel(const Scalar * __restrict__ v, const RealType * __restrict__ s, Scalar * __restrict__ res, int size) {
  RealType scalar = *s;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    res[i] = v[i] / scalar;
  }
}

template <>
__global__ void divideByRealKernel<cuComplex, float>(const cuComplex * __restrict__ v, const float * __restrict__ s, cuComplex * __restrict__ res, int size) {
  float scalar = *s;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    res[i].x = v[i].x / scalar;
    res[i].y = v[i].y / scalar;
  }
}

template <>
__global__ void divideByRealKernel<cuDoubleComplex, double>(const cuDoubleComplex * __restrict__ v, const double * __restrict__ s, cuDoubleComplex * __restrict__ res, int size) {
  double scalar = *s;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    res[i].x = v[i].x / scalar;
    res[i].y = v[i].y / scalar;
  }
}


template <typename Scalar>
__global__ void eigshNormalizeKernel(Scalar * __restrict__ col, Scalar * __restrict__ v, int n,
                                     const Scalar * __restrict__ u ,const Scalar * __restrict__ beta)
{
  Scalar b = *beta;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    Scalar res = u[i] / b;
    col[i] = res;
    v[i] = res;
  }
}

template <>
__global__ void eigshNormalizeKernel<cuComplex>(cuComplex * __restrict__ col, cuComplex * __restrict__ v, int n,
                                                const cuComplex * __restrict__ u ,const cuComplex * __restrict__ beta)
{
  float b = beta->x;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    cuComplex res = make_cuComplex(u[i].x / b, u[i].y / b);
    col[i] = res;
    v[i] = res;
  }
}

template <>
__global__ void eigshNormalizeKernel<cuDoubleComplex>(cuDoubleComplex * __restrict__ col, cuDoubleComplex * __restrict__ v, int n,
                                                      const cuDoubleComplex * __restrict__ u ,const cuDoubleComplex * __restrict__ beta)
{
  double b = beta->x;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    cuDoubleComplex res = make_cuDoubleComplex(u[i].x / b, u[i].y / b);
    col[i] = res;
    v[i] = res;
  }
}

void _permute(float *v, const int *perm, void *buffer, int size)
{
  using Scalar = float;
  permuteKernel<Scalar><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, perm, (Scalar *) buffer, size);
  CHECK_CUDA( cudaMemcpy(v, buffer, size * sizeof(Scalar), cudaMemcpyDeviceToDevice) );
}

void _permute(double *v, const int *perm, void *buffer, int size)
{
  using Scalar = double;
  permuteKernel<Scalar><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, perm, (Scalar *) buffer, size);
  CHECK_CUDA( cudaMemcpy(v, buffer, size * sizeof(Scalar), cudaMemcpyDeviceToDevice) );
}

void _permute(cuComplex *v, const int *perm, void *buffer, int size)
{
  using Scalar = cuComplex;
  permuteKernel<Scalar><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, perm, (Scalar *) buffer, size);
  CHECK_CUDA( cudaMemcpy(v, buffer, size * sizeof(Scalar), cudaMemcpyDeviceToDevice) );
}

void _permute(cuDoubleComplex *v, const int *perm, void *buffer, int size)
{
  using Scalar = cuDoubleComplex;
  permuteKernel<Scalar><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, perm, (Scalar *) buffer, size);
  CHECK_CUDA( cudaMemcpy(v, buffer, size * sizeof(Scalar), cudaMemcpyDeviceToDevice) );
}

void addInPlace(float * __restrict__ lhs, float * __restrict__ rhs) {
  addInPlaceKernel<float><<<1, 1>>>(lhs, rhs);
}

void addInPlace(double * __restrict__ lhs, double * __restrict__ rhs) {
  addInPlaceKernel<double><<<1, 1>>>(lhs, rhs);
}

void addInPlace(cuComplex * __restrict__ lhs, cuComplex * __restrict__ rhs) {
  addInPlaceKernel<cuComplex><<<1, 1>>>(lhs, rhs);
}

void addInPlace(cuDoubleComplex * __restrict__ lhs, cuDoubleComplex * __restrict__ rhs) {
  addInPlaceKernel<cuDoubleComplex><<<1, 1>>>(lhs, rhs);
}

void eigshNormalize(float * __restrict__ col, float * __restrict__ v, int n,
                    const float * __restrict__ u ,const float * __restrict__ beta)
{
  eigshNormalizeKernel<float><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(col, v, n, u, beta);
}

void eigshNormalize(double * __restrict__ col, double * __restrict__ v, int n,
                    const double * __restrict__ u ,const double * __restrict__ beta)
{
  eigshNormalizeKernel<double><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(col, v, n, u, beta);
}

void eigshNormalize(cuComplex * __restrict__ col, cuComplex * __restrict__ v, int n,
                    const cuComplex * __restrict__ u ,const cuComplex * __restrict__ beta)
{
  eigshNormalizeKernel<cuComplex><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(col, v, n, u, beta);
}

void eigshNormalize(cuDoubleComplex * __restrict__ col, cuDoubleComplex * __restrict__ v, int n,
                    const cuDoubleComplex * __restrict__ u ,const cuDoubleComplex * __restrict__ beta)
{
  eigshNormalizeKernel<cuDoubleComplex><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(col, v, n, u, beta);
}

void _divideByReal(const float * __restrict__ v, const float * __restrict__ s, float * __restrict__ res, int size) {
  divideByRealKernel<float, float><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, s, res, size);
}

void _divideByReal(const double * __restrict__ v, const double * __restrict__ s, double * __restrict__ res, int size) {
  divideByRealKernel<double, double><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, s, res, size);
}

void _divideByReal(const cuComplex * __restrict__ v, const float * __restrict__ s, cuComplex * __restrict__ res, int size) {
  divideByRealKernel<cuComplex, float><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, s, res, size);
}

void _divideByReal(const cuDoubleComplex * __restrict__ v, const double * __restrict__ s, cuDoubleComplex * __restrict__ res, int size) {
  divideByRealKernel<cuDoubleComplex, double><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, s, res, size);
}

} // Namespace GPU
