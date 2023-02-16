namespace GPU
{
inline cusparseStatus_t cusparseCsrsm2_bufferSize(cusparseHandle_t         handle,
                                                  cusparseOperation_t      transA,
                                                  cusparseOperation_t      transB,
                                                  int                      m,
                                                  int                      nrhs,
                                                  int                      nnz,
                                                  const float*             alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  float*                   csrValA,
                                                  const int*               csrRowPtrA,
                                                  const int*               csrColIndA,
                                                  const float*             B,
                                                  int                      ldb,
                                                  csrsm2Info_t             info,
                                                  cusparseSolvePolicy_t    policy,
                                                  size_t*                  pBufSize,
                                                  int                      algo = 1)
{
  return cusparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBufSize);
}

inline cusparseStatus_t cusparseCsrsm2_bufferSize(cusparseHandle_t         handle,
                                                  cusparseOperation_t      transA,
                                                  cusparseOperation_t      transB,
                                                  int                      m,
                                                  int                      nrhs,
                                                  int                      nnz,
                                                  const double*            alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  double*                  csrValA,
                                                  const int*               csrRowPtrA,
                                                  const int*               csrColIndA,
                                                  const double*            B,
                                                  int                      ldb,
                                                  csrsm2Info_t             info,
                                                  cusparseSolvePolicy_t    policy,
                                                  size_t*                  pBufSize,
                                                  int                      algo = 1)
{
  return cusparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBufSize);
}

inline cusparseStatus_t cusparseCsrsm2_bufferSize(cusparseHandle_t         handle,
                                                  cusparseOperation_t      transA,
                                                  cusparseOperation_t      transB,
                                                  int                      m,
                                                  int                      nrhs,
                                                  int                      nnz,
                                                  const cuComplex*         alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  cuComplex*               csrValA,
                                                  const int*               csrRowPtrA,
                                                  const int*               csrColIndA,
                                                  const cuComplex*         B,
                                                  int                      ldb,
                                                  csrsm2Info_t             info,
                                                  cusparseSolvePolicy_t    policy,
                                                  size_t*                  pBufSize,
                                                  int                      algo = 1)
{
  return cusparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBufSize);
}

inline cusparseStatus_t cusparseCsrsm2_bufferSize(cusparseHandle_t         handle,
                                                  cusparseOperation_t      transA,
                                                  cusparseOperation_t      transB,
                                                  int                      m,
                                                  int                      nrhs,
                                                  int                      nnz,
                                                  const cuDoubleComplex*   alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  cuDoubleComplex*         csrValA,
                                                  const int*               csrRowPtrA,
                                                  const int*               csrColIndA,
                                                  const cuDoubleComplex*   B,
                                                  int                      ldb,
                                                  csrsm2Info_t             info,
                                                  cusparseSolvePolicy_t    policy,
                                                  size_t*                  pBufSize,
                                                  int                      algo = 1)
{
  return cusparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBufSize);
}

inline cusparseStatus_t cusparseCsrsm2_analysis(cusparseHandle_t         handle,
                                                cusparseOperation_t      transA,
                                                cusparseOperation_t      transB,
                                                int                      m,
                                                int                      nrhs,
                                                int                      nnz,
                                                const float*             alpha,
                                                const cusparseMatDescr_t descrA,
                                                float*                   csrValA,
                                                const int*               csrRowPtrA,
                                                const int*               csrColIndA,
                                                const float*             B,
                                                int                      ldb,
                                                csrsm2Info_t             info,
                                                cusparseSolvePolicy_t    policy,
                                                void*                    pBuffer,
                                                int                      algo = 1)
{
  return cusparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_analysis(cusparseHandle_t         handle,
                                                cusparseOperation_t      transA,
                                                cusparseOperation_t      transB,
                                                int                      m,
                                                int                      nrhs,
                                                int                      nnz,
                                                const double*            alpha,
                                                const cusparseMatDescr_t descrA,
                                                double*                  csrValA,
                                                const int*               csrRowPtrA,
                                                const int*               csrColIndA,
                                                const double*            B,
                                                int                      ldb,
                                                csrsm2Info_t             info,
                                                cusparseSolvePolicy_t    policy,
                                                void*                    pBuffer,
                                                int                      algo = 1)
{
  return cusparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_analysis(cusparseHandle_t         handle,
                                                cusparseOperation_t      transA,
                                                cusparseOperation_t      transB,
                                                int                      m,
                                                int                      nrhs,
                                                int                      nnz,
                                                const cuComplex*         alpha,
                                                const cusparseMatDescr_t descrA,
                                                cuComplex*               csrValA,
                                                const int*               csrRowPtrA,
                                                const int*               csrColIndA,
                                                const cuComplex*         B,
                                                int                      ldb,
                                                csrsm2Info_t             info,
                                                cusparseSolvePolicy_t    policy,
                                                void*                    pBuffer,
                                                int                      algo = 1)
{
  return cusparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_analysis(cusparseHandle_t         handle,
                                                cusparseOperation_t      transA,
                                                cusparseOperation_t      transB,
                                                int                      m,
                                                int                      nrhs,
                                                int                      nnz,
                                                const cuDoubleComplex*   alpha,
                                                const cusparseMatDescr_t descrA,
                                                cuDoubleComplex*         csrValA,
                                                const int*               csrRowPtrA,
                                                const int*               csrColIndA,
                                                const cuDoubleComplex*   B,
                                                int                      ldb,
                                                csrsm2Info_t             info,
                                                cusparseSolvePolicy_t    policy,
                                                void*                    pBuffer,
                                                int                      algo = 1)
{
  return cusparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_solve(cusparseHandle_t         handle,
                                             cusparseOperation_t      transA,
                                             cusparseOperation_t      transB,
                                             int                      m,
                                             int                      nrhs,
                                             int                      nnz,
                                             const float*             alpha,
                                             const cusparseMatDescr_t descrA,
                                             float*                   csrValA,
                                             const int*               csrRowPtrA,
                                             const int*               csrColIndA,
                                             float*                   B,
                                             int                      ldb,
                                             csrsm2Info_t             info,
                                             cusparseSolvePolicy_t    policy,
                                             void*                    pBuffer,
                                             int                      algo = 1)
{
  return cusparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_solve(cusparseHandle_t         handle,
                                             cusparseOperation_t      transA,
                                             cusparseOperation_t      transB,
                                             int                      m,
                                             int                      nrhs,
                                             int                      nnz,
                                             const double*            alpha,
                                             const cusparseMatDescr_t descrA,
                                             double*                  csrValA,
                                             const int*               csrRowPtrA,
                                             const int*               csrColIndA,
                                             double*                  B,
                                             int                      ldb,
                                             csrsm2Info_t             info,
                                             cusparseSolvePolicy_t    policy,
                                             void*                    pBuffer,
                                             int                      algo = 1)
{
  return cusparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_solve(cusparseHandle_t         handle,
                                             cusparseOperation_t      transA,
                                             cusparseOperation_t      transB,
                                             int                      m,
                                             int                      nrhs,
                                             int                      nnz,
                                             const cuComplex*         alpha,
                                             const cusparseMatDescr_t descrA,
                                             cuComplex*               csrValA,
                                             const int*               csrRowPtrA,
                                             const int*               csrColIndA,
                                             cuComplex*               B,
                                             int                      ldb,
                                             csrsm2Info_t             info,
                                             cusparseSolvePolicy_t    policy,
                                             void*                    pBuffer,
                                             int                      algo = 1)
{
  return cusparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}

inline cusparseStatus_t cusparseCsrsm2_solve(cusparseHandle_t         handle,
                                             cusparseOperation_t      transA,
                                             cusparseOperation_t      transB,
                                             int                      m,
                                             int                      nrhs,
                                             int                      nnz,
                                             const cuDoubleComplex*   alpha,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex*         csrValA,
                                             const int*               csrRowPtrA,
                                             const int*               csrColIndA,
                                             cuDoubleComplex*         B,
                                             int                      ldb,
                                             csrsm2Info_t             info,
                                             cusparseSolvePolicy_t    policy,
                                             void*                    pBuffer,
                                             int                      algo = 1)
{
  return cusparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, pBuffer);
}
} // namepace GPU
