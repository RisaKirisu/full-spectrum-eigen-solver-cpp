#pragma once

namespace GPU {

template <typename Scalar>
__global__ void permuteKernel(const Scalar* __restrict__ v, const int* __restrict__ perm, Scalar* __restrict__ dst, int size);

template <typename Scalar>
__global__ void addInPlaceKernel(Scalar * __restrict__ lhs, Scalar * __restrict__ rhs);

template <typename Scalar, typename RealType>
__global__ void divideByRealKernel(const Scalar * __restrict__ v, const RealType * __restrict__ s, int size);

template <typename Scalar>
__global__ void eigshNormalizeKernel(Scalar * __restrict__ col, Scalar * __restrict__ v, int n,
                                const Scalar * __restrict__ u ,const Scalar * __restrict__ beta);

void _permute(float *v, const int *perm, void *buffer, int size);

void _permute(double *v, const int *perm, void *buffer, int size);

void _permute(cuComplex *v, const int *perm, void *buffer, int size);

void _permute(cuDoubleComplex *v, const int *perm, void *buffer, int size);

void addInPlace(float * __restrict__ lhs, float * __restrict__ rhs);

void addInPlace(double * __restrict__ lhs, double * __restrict__ rhs);

void addInPlace(cuComplex * __restrict__ lhs, cuComplex * __restrict__ rhs);

void addInPlace(cuDoubleComplex * __restrict__ lhs, cuDoubleComplex * __restrict__ rhs);

void eigshNormalize(float * __restrict__ col, float * __restrict__ v, int n,
                    const float * __restrict__ u ,const float * __restrict__ beta);

void eigshNormalize(double * __restrict__ col, double * __restrict__ v, int n,
                    const double * __restrict__ u ,const double * __restrict__ beta);

void eigshNormalize(cuComplex * __restrict__ col, cuComplex * __restrict__ v, int n,
                    const cuComplex * __restrict__ u ,const cuComplex * __restrict__ beta);

void eigshNormalize(cuDoubleComplex * __restrict__ col, cuDoubleComplex * __restrict__ v, int n,
                    const cuDoubleComplex * __restrict__ u ,const cuDoubleComplex * __restrict__ beta);

void _divideByReal(const float * __restrict__ v, const float * __restrict__ s, float * __restrict__ res, int size);

void _divideByReal(const double * __restrict__ v, const double * __restrict__ s, double * __restrict__ res, int size);

void _divideByReal(const cuComplex * __restrict__ v, const float * __restrict__ s, cuComplex * __restrict__ res, int size);

void _divideByReal(const cuDoubleComplex * __restrict__ v, const double * __restrict__ s, cuDoubleComplex * __restrict__ res, int size);
}