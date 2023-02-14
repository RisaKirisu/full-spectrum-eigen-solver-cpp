#pragma once

namespace GPU {

template <typename Scalar>
__global__ void permuteKernel(const Scalar* __restrict__ v, const int* __restrict__ perm, Scalar* __restrict__ dst, int size);

template <typename Scalar>
__global__ void addInPlaceKernel(Scalar * __restrict__ lhs, Scalar * __restrict__ rhs);

template <typename Scalar, typename RealType>
__global__ void divideByRealKernel(Scalar * __restrict__ v, const RealType * __restrict__ s, int size);

template <typename Scalar>
__global__ void eigshNormalizeKernel(Scalar * __restrict__ col, Scalar * __restrict__ v, int n,
                                const Scalar * __restrict__ u ,const Scalar * __restrict__ beta);

template <typename Scalar>
void _permute(Scalar *v, const int *perm, void *buffer, int size);

template <typename Scalar>
__global__ void addInPlace(Scalar * __restrict__ lhs, Scalar * __restrict__ rhs);

template <typename Scalar>
void eigshNormalize(Scalar * __restrict__ col, Scalar * __restrict__ v, int n,
                    const Scalar * __restrict__ u ,const Scalar * __restrict__ beta);

template <typename Scalar, typename RealType>
void _divideByReal(const Scalar * __restrict__ v, const RealType * __restrict__ s, Scalar * __restrict__ res, int size);
}