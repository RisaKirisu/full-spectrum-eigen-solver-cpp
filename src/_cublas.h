namespace GPU {



template <typename T>
inline cudaDataType dtype();

template <>
inline cudaDataType dtype<float>() {
  return CUDA_R_32F;
}

template <>
inline cudaDataType dtype<double>() {
  return CUDA_R_64F;
}

template <>
inline cudaDataType dtype<std::complex<float>>() {
  return CUDA_C_32F;
}

template <>
inline cudaDataType dtype<std::complex<double>>() {
  return CUDA_C_64F;
}

template <typename T>
struct real_type {
  using RealType = T;
  using CudaType = T;
};

template <>
struct real_type<std::complex<float>> {
  using RealType = float;
  using CudaType = cuComplex;
};

template <>
struct real_type<std::complex<double>> {
  using RealType = double;
  using CudaType = cuDoubleComplex;
};

inline cublasStatus_t _gemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float *alpha,
                           const float *A,     int lda,
                           const float *x,     int incx,
                           const float *beta,
                           float *y,           int incy)
{
  return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t _gemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double *alpha,
                           const double *A,     int lda,
                           const double *x,     int incx,
                           const double *beta,
                           double *y,           int incy)
{
  return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t _gemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const cuComplex *alpha,
                           const cuComplex *A,     int lda,
                           const cuComplex *x,     int incx,
                           const cuComplex *beta,
                           cuComplex *y,           int incy)
{
  return cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t _gemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A,     int lda,
                           const cuDoubleComplex *x,     int incx,
                           const cuDoubleComplex *beta,
                           cuDoubleComplex *y,           int incy)
{
  return cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t _axpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy)
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t _axpy(cublasHandle_t handle, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           double                 *y, int incy)
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t _axpy(cublasHandle_t handle, int n,
                           const cuComplex       *alpha,
                           const cuComplex       *x, int incx,
                           cuComplex             *y, int incy)
{
  return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t _axpy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex       *y, int incy)
{
  return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t _dotc (cublasHandle_t handle, int n,
                     const float    *x, int incx,
                     const float    *y, int incy,
                     float          *result)
{ 
  return cublasSdot (handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t _dotc (cublasHandle_t handle, int n,
                     const double    *x, int incx,
                     const double    *y, int incy,
                     double          *result)
{
  return cublasDdot (handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t _dotc (cublasHandle_t handle, int n,
                     const cuComplex *x, int incx,
                     const cuComplex *y, int incy,
                     cuComplex       *result)
{ 
  return cublasCdotc (handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t _dotc (cublasHandle_t handle, int n,
                     const cuDoubleComplex *x, int incx,
                     const cuDoubleComplex *y, int incy,
                     cuDoubleComplex       *result)
{
  return cublasZdotc (handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t _nrm2(cublasHandle_t handle, int n,
                     const float *x, int incx, float *result)
{
  return cublasSnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t _nrm2(cublasHandle_t handle, int n,
                     const double *x, int incx, double *result)
{
  return cublasDnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t _nrm2(cublasHandle_t handle, int n,
                     const cuComplex *x, int incx, float *result)
{
  return cublasScnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t _nrm2(cublasHandle_t handle, int n,
                     const cuDoubleComplex *x, int incx, double *result)
{
  return cublasDznrm2(handle, n, x, incx, result);
}

inline cublasStatus_t _gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float   *alpha,
                           const float   *A, int lda,
                           const float   *B, int ldb,
                           const float   *beta,
                           float         *C, int ldc, int cc)
{
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t _gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double   *alpha,
                           const double   *A, int lda,
                           const double   *B, int ldb,
                           const double   *beta,
                           double         *C, int ldc, int cc)
{
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t _gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex   *alpha,
                           const cuComplex   *A, int lda,
                           const cuComplex   *B, int ldb,
                           const cuComplex   *beta,
                           cuComplex         *C, int ldc, int cc)
{
  if (cc >= 5)
    return cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  else
    return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t _gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex   *alpha,
                           const cuDoubleComplex   *A, int lda,
                           const cuDoubleComplex   *B, int ldb,
                           const cuDoubleComplex   *beta,
                           cuDoubleComplex         *C, int ldc, int cc)
{
  if (cc >= 5)
    return cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  else
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const float           *alpha,
                           float           *x, int incx)
{
  return cublasSscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const double          *alpha,
                           double          *x, int incx)
{
  return cublasDscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const cuComplex       *alpha,
                           cuComplex       *x, int incx)
{
  return cublasCscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const float           *alpha,
                           cuComplex       *x, int incx)
{
  return cublasCsscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           cuDoubleComplex *x, int incx)
{
  return cublasZscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _scal(cublasHandle_t handle, int n,
                           const double          *alpha,
                           cuDoubleComplex *x, int incx)
{
  return cublasZdscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t _copy(cublasHandle_t handle, int n,
                            const float     *x, int incx,
                            float           *y, int incy)
{
  return cublasScopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t _copy(cublasHandle_t handle, int n,
                            const double    *x, int incx,
                            double          *y, int incy)
{
  return cublasDcopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t _copy(cublasHandle_t handle, int n,
                            const cuComplex *x, int incx,
                            cuComplex       *y, int incy)
{
  return cublasCcopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t _copy(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            cuDoubleComplex*y, int incy)
{
  return cublasZcopy(handle, n, x, incx, y, incy);
}

}
