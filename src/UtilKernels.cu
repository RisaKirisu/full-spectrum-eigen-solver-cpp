#include <cuComplex.h>

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
__global__ void divideByRealKernel(Scalar * __restrict__ v, const RealType * __restrict__ s, Scalar * __restrict__ res, int size) {
  RealType scalar = *s;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    v[i] /= scalar;
  }
}

template <>
__global__ void divideByRealKernel<cuComplex, float>(cuComplex * __restrict__ v, const float * __restrict__ s, cuComplex * __restrict__ res, int size) {
  float scalar = *s;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    res[i].x = v[i].x / scalar;
    res[i].y = v[i].y / scalar;
  }
}

template <>
__global__ void divideByRealKernel<cuDoubleComplex, double>(cuDoubleComplex * __restrict__ v, const double * __restrict__ s, cuDoubleComplex * __restrict__ res, int size) {
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

template <typename Scalar>
void _permute(Scalar *v, const int *perm, void *buffer, int size)
{
  permuteKernel<Scalar><<<std::min((int) std::ceil(size / 512.0f), 80), 512>>>(v, perm, (Scalar *) buffer, size);
  CHECK_CUDA( cudaMemcpy(v, buffer, size * sizeof(Scalar), cudaMemcpyDeviceToDevice) );
}

template <typename Scalar>
void addInPlace(Scalar * __restrict__ lhs, Scalar * __restrict__ rhs) {
  addInPlaceKernel<Scalar><<<1, 1>>>(lhs, rhs);
}

template <typename Scalar>
inline void eigshNormalize(Scalar * __restrict__ col, Scalar * __restrict__ v, int n,
                           const Scalar * __restrict__ u ,const Scalar * __restrict__ beta)
{
  eigshNormalizeKernel<Scalar><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(col, v, n, u, beta);
}

template <typename Scalar, typename RealType>
void _divideByReal(Scalar * __restrict__ v, const RealType * __restrict__ s, Scalar * __restrict__ res, int size) {
  divideByRealKernel<Scalar, RealType><<<std::min((int) std::ceil(n / 512.0f), 80), 512>>>(v, s, res, size);
}

} // Namespace GPU
