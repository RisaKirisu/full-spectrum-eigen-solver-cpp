#pragma once

namespace GPU {
template <typename Scalar>
class cusparseLinearOperator {
  public:
    // Let op() be the operator. Performs out = alpha * op(rhs) + beta * out
    virtual void matvec(cusparseHandle_t handle,
                        const cuVector<Scalar> &rhs,
                        cuVector<Scalar> &out) = 0;

    // Number of rows
    virtual int rows() const = 0;
    // Number of columns
    virtual int cols() const = 0;
    // Number of non-zero elements
    virtual int nnz() const = 0;
};

template <typename Scalar>
class cuparseCsrMatrix : public cusparseLinearOperator<Scalar> {
  public:
    cuparseCsrMatrix() {}

    // Construct cusparseCsrMatrix from a generic Eigen sparse matrix. The input must be compressed.
    cuparseCsrMatrix(const Eigen::SparseMatrix<Scalar> &matrix) {
      setMatrix(matrix);
    }

    /*
     *cuparseCsrMatrix(cudaStream_t stream, const Eigen::SparseMatrix<Scalar> &matrix) {
     *m_stream = stream;
     *setMatrix(matrix);
     *}
    */

    // Construct cusparseCsrMatrix directly from c arrays.
    cuparseCsrMatrix(const void *data, const void *rowPtr, const void *colIdx, const int nnz, const int nrow, const int ncol) {
      m_nnz = nnz;
      m_rows = nrow;
      m_cols = ncol;

      _setMatrix(rowPtr, colIdx, data, cudaMemcpyHostToDevice);
    }

    // Copy constructor
    cuparseCsrMatrix(const cuparseCsrMatrix<Scalar> &rhs) {
      _copy(rhs);
    }

    // Destructor
    ~cuparseCsrMatrix() {
      _destroy();
    }

    // Copy assigment operator
    cuparseCsrMatrix<Scalar>& operator=(const cuparseCsrMatrix<Scalar> &rhs) {
      if (this != &rhs) {
        _copy(rhs);
      }
      return *this;
    }

    cuparseCsrMatrix<Scalar>& operator=(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &rhs) {
      setMatrix(rhs);
      return *this;
    }

    // Construct a cusparse csr matrix on GPU from an Eigen sparse matrix
    void setMatrix(const Eigen::SparseMatrix<Scalar> &matrix) {
      /* If a cusparse matrix already exist, destroy it and create a new one*/
      _destroy();

      m_nnz = matrix.nonZeros();
      m_rows = matrix.rows();
      m_cols = matrix.cols();
      
      // Need to convert to CSR format if matrix is not already in csr format.
      if (matrix.IsRowMajor) {
        _setMatrix(matrix.outerIndexPtr(), matrix.innerIndexPtr(), matrix.valuePtr(), cudaMemcpyHostToDevice);
      } else {
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> csr(matrix);
        _setMatrix(csr.outerIndexPtr(), csr.innerIndexPtr(), csr.valuePtr(), cudaMemcpyHostToDevice);
      }
      
    }

    void spMv(cusparseHandle_t handle, const cuVector<Scalar> &rhs, const Scalar alpha, const Scalar beta, cuVector<Scalar> &out) {
      if (m_descr == NULL) {
        fprintf(stderr, "spMv: matrix is not initialized.\n");
        exit(1);
      }
      if (cols() != rhs.size()) {
        fprintf(stderr, "spMv: shape mismatch. Multiplying (%d, %d) and (%d)\n", rows(), cols(), rhs.size());
        exit(1);
      }

      // Resize output vector to correcto size
      if (rows() != out.size()) {
        out.resize(rows());
      }

      // Create dense vector descriptor
      cusparseDnVecDescr_t descrRhs, descrOut;
      CHECK_CUSPARSE( cusparseCreateDnVec(&descrOut, out.size(), out.data(), dtype<Scalar>()) );
      CHECK_CUSPARSE( cusparseCreateDnVec(&descrRhs, rhs.size(), rhs.data(), dtype<Scalar>()) );

      // Set pointer mode
      cusparsePointerMode_t pointerMode;
      CHECK_CUSPARSE( cusparseGetPointerMode(handle, &pointerMode) );
      CHECK_CUSPARSE( cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST) );

      // Allocate work buffer isn't it's not already allocated.
      if (m_workBuf == nullptr) {
        size_t bufSz = 0;
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, m_descr, descrRhs, &beta, descrOut,
                                dtype<Scalar>(), CUSPARSE_SPMV_CSR_ALG1,
                                &bufSz) );
        CHECK_CUDA( cudaMalloc((void **) &m_workBuf, (size_t) (1.2 * bufSz)) );
      }
      
      // Calculate
      CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, m_descr, descrRhs, &beta, descrOut,
                                   dtype<Scalar>(), CUSPARSE_SPMV_CSR_ALG1,
                                   m_workBuf) );

      // Restore pointer mode
      CHECK_CUSPARSE( cusparseSetPointerMode(handle, pointerMode) );

      // Destroy vector descriptors
      CHECK_CUSPARSE( cusparseDestroyDnVec(descrRhs) );
      CHECK_CUSPARSE( cusparseDestroyDnVec(descrOut) );
    }

    // Let A be self. Performs out = A * rhs
    void matvec(cusparseHandle_t handle, const cuVector<Scalar> &rhs, cuVector<Scalar> &out) {
      Scalar alpha(1), beta(0);
      spMv(handle, rhs, alpha, beta, out);
    }

    // Number of rows
    int rows() const { return m_rows; }

    // Number of cols
    int cols() const { return m_cols; }

    // Number of non-zero elements
    int nnz() const { return m_nnz; }

    /* Pointer to row pointer array.
     *Note: the pointer points to an address in GPU memory.
    */ 
    void *rowPosPtr() const { return m_dCsrRowPtr; }

    /* Pointer to column indices array.
     *Note: the pointer points to an address in GPU memory.
    */ 
    void *colIdxPtr() const { return m_dCsrColIdx; }

    /* Pointer to non-zero element array.
     *Note: the pointer points to an address in GPU memory.
    */ 
    void *valuePtr() const { return m_dCsrVal; }

    // The cusparse sparse matrix descriptor of this matrix.
    cusparseSpMatDescr_t descriptor() const { return m_descr; }

  private:
    cudaStream_t m_stream = NULL;
    cusparseSpMatDescr_t m_descr = NULL;
    void *m_dCsrRowPtr = nullptr;
    void *m_dCsrColIdx = nullptr;
    void *m_dCsrVal = nullptr;
    void *m_workBuf = nullptr;
    int m_nnz = 0;
    int m_rows = 0;
    int m_cols = 0;

    void _destroy() {
      if (!m_descr) {
        return;
      }
      CHECK_CUSPARSE( cusparseDestroySpMat(m_descr) );
      CHECK_CUDA( cudaFree(m_dCsrRowPtr) );
      CHECK_CUDA( cudaFree(m_dCsrColIdx) );
      CHECK_CUDA( cudaFree(m_dCsrVal) );

      if (m_workBuf) {
        CHECK_CUDA( cudaFree(m_workBuf) );
        m_workBuf = nullptr;
      }

      m_descr = NULL;
      m_nnz = 0;
      m_rows = 0;
      m_cols = 0;
    }

    void _copy(const cuparseCsrMatrix<Scalar> &rhs) {
      _destroy();
      m_stream = rhs.m_stream;
      m_nnz = rhs.m_nnz;
      m_rows = rhs.m_rows;
      m_cols = rhs.m_cols;

      _setMatrix(rhs.m_dCsrRowPtr, rhs.m_dCsrColIdx, rhs.m_dCsrVal, cudaMemcpyDeviceToDevice);
    }

    void _setMatrix(const void *rowPtr, const void *colIdx, const void *values, cudaMemcpyKind direction) {
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrRowPtr, (rows() + 1) * sizeof(int)) );
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrColIdx, nnz() * sizeof(int))        );
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrVal,    nnz() * sizeof(Scalar))     );

      CHECK_CUDA( cudaMemcpy(m_dCsrRowPtr,  rowPtr, (rows() + 1) * sizeof(int), direction) );
      CHECK_CUDA( cudaMemcpy(m_dCsrColIdx,  colIdx, nnz() * sizeof(int),        direction) );
      CHECK_CUDA( cudaMemcpy(m_dCsrVal,     values, nnz() * sizeof(Scalar),     direction) );

      CHECK_CUSPARSE( cusparseCreateCsr(&m_descr, rows(), cols(), nnz(),
                                        m_dCsrRowPtr, m_dCsrColIdx, m_dCsrVal,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        dtype<Scalar>()) );
    }

};

template <typename Scalar>
class cusparseLU : public cusparseLinearOperator<Scalar> {
  public:
    // Construct cusparseLU from solved Eigen SparseLU objects.
    cusparseLU(Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> &lu)
      : m_perm_r(lu.rowsPermutation().indices()), m_perm_cInv(lu.colsPermutation().inverse().eval().indices())
    {
      Eigen::SparseMatrix<Scalar, Eigen::RowMajor> lcsr, ucsr;
      lu.getCsrLU(lcsr, ucsr);
      m_L = lcsr;
      m_U = ucsr;
      _setMatAttr();
    }

    // Construct cusparseLU from Eigen SparseMatrix
    cusparseLU(Eigen::SparseMatrix<Scalar> &L, Eigen::SparseMatrix<Scalar> &U,
               Eigen::PermutationMatrix<Eigen::Dynamic> perm_r, Eigen::PermutationMatrix<Eigen::Dynamic> perm_c) 
      : m_L(L), m_U(U), m_perm_r(perm_r.indices()), m_perm_cInv(perm_c.inverse().eval().indices())
    {
      _setMatAttr();
    }

    cusparseLU(const void *Ldata, const void *LcolIdx, const void *LrowPtr,
               const void *Udata, const void *UcolIdx, const void *UrowPtr,
               const void *perm_r, const void *perm_cInv,
               const int nnzL, const int nnzU, const int rows)
      : m_L(Ldata, LrowPtr, LcolIdx, nnzL, rows, rows), m_U(Udata, UrowPtr, UcolIdx, nnzU, rows, rows),
        m_perm_r(perm_r, rows, false), m_perm_cInv(perm_cInv, rows, false)
    {
      _setMatAttr();
    }

    cusparseLU(const cusparseLU<Scalar> &rhs) {
      _copy(rhs);
    }

    ~cusparseLU() {
      _destroy();
    }

    cusparseLU<Scalar>& operator=(const cusparseLU<Scalar> &rhs) {
      if (this != &rhs) {
        _copy(rhs);
      }
      return *this;
    }

    // Let Pr * A * Pc.T = L * U. This solves a system of linear equation: A * out = alpha * rhs
    void solve(cusparseHandle_t handle, const cuVector<Scalar> &rhs, const Scalar alpha, cuVector<Scalar> &out) {
      if (m_L.descriptor() == NULL) {
        fprintf(stderr, "Solve: matrix is not initialized.\n");
        exit(1);
      }
      if (cols() != rhs.size()) {
        fprintf(stderr, "Solve: shape mismatch. Solving system (%d, %d) with rhs (%d)\n", rows(), cols(), rhs.size());
        exit(1);
      }

      // Create dense vector descriptor
      cusparseDnVecDescr_t descrOut;
      CHECK_CUSPARSE( cusparseCreateDnVec(&descrOut, out.size(), out.data(), dtype<Scalar>()) );

      // Set pointer mode
      cusparsePointerMode_t pointerMode;
      CHECK_CUSPARSE( cusparseGetPointerMode(handle, &pointerMode) );
      CHECK_CUSPARSE( cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST) );

      // Create SpSV descriptor if not already created
      if (m_spsvDescrL == NULL) {
        CHECK_CUSPARSE( cusparseSpSV_createDescr(&m_spsvDescrL) );
        CHECK_CUSPARSE( cusparseSpSV_createDescr(&m_spsvDescrU) );
      }

      // Allocate work buffer and perform analysis if not already allocated
      if (m_dataBuf == nullptr) {
        size_t bufSzL = 0;
        size_t bufSzU = 0;
        CHECK_CUSPARSE( cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, m_L.descriptor(), descrOut, descrOut,
                                                dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                                m_spsvDescrL, &bufSzL) );
        CHECK_CUSPARSE( cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, m_U.descriptor(), descrOut, descrOut,
                                                dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                                m_spsvDescrU, &bufSzU) );
        CHECK_CUDA( cudaMalloc((void **) m_dataBuf, std::max((size_t) rows(), std::max(bufSzL, bufSzU))) );

        CHECK_CUSPARSE( cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, m_L.descriptor(), descrOut, descrOut,
                                              dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                              m_spsvDescrL, m_dataBuf) );
        CHECK_CUSPARSE( cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, m_U.descriptor(), descrOut, descrOut,
                                              dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                              m_spsvDescrU, m_dataBuf) );
      }
      
      // Apply row permutation
      out = rhs;
      out.permute(m_perm_r, m_dataBuf);

      // Solve L
      CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, m_L.descriptor(),descrOut, descrOut,
                                         dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                         m_spsvDescrL) );
      // Solve U
      Scalar one(1);
      CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, m_U.descriptor(), descrOut, descrOut,
                                         dtype<Scalar>(), CUSPARSE_SPSV_ALG_DEFAULT,
                                         m_spsvDescrU) );

      // Apply inverse of column permutation
      out.permute(m_perm_cInv, m_dataBuf);
    }

    // Let Pr * A * Pc.T = L * U. This calculate: out = A^-1 * rhs
    void matvec(cusparseHandle_t handle, const cuVector<Scalar> &rhs, cuVector<Scalar> &out) {
      Scalar alpha(1);
      solve(handle, rhs, alpha, out);
    }

    // Number of rows
    int rows() const { return m_L.rows(); }

    // Number of columns
    int cols() const { return m_L.rows(); }

    // Number of non-zero elements in total
    int nnz() const { return m_L.nnz() + m_U.nnz(); }

    // Number of non-zero elements in L
    int nnzL() const { return m_L.nnz(); }

    // Number of non-zero elements in U
    int nnzU() const { return m_U.nnz(); }

  private:
    cuparseCsrMatrix<Scalar> m_L, m_U;
    cuVector<int> m_perm_r, m_perm_cInv;
    cusparseSpSVDescr_t m_spsvDescrL = NULL;
    cusparseSpSVDescr_t m_spsvDescrU = NULL;

    void *m_dataBuf = nullptr;

    void _destroy() {
      if (m_dataBuf) {
        CHECK_CUDA( cudaFree(m_dataBuf) );
        m_dataBuf = nullptr;
      }

      if (m_spsvDescrL) {
        CHECK_CUSPARSE( cusparseSpSV_destroyDescr(m_spsvDescrL) );
        CHECK_CUSPARSE( cusparseSpSV_destroyDescr(m_spsvDescrU) );
        m_spsvDescrL = NULL;
        m_spsvDescrU = NULL;
      }
    }

    void _copy(const cusparseLU<Scalar> &rhs) {
      _destroy();
      m_L = rhs.m_L;
      m_U = rhs.m_U;
      m_perm_r = rhs.m_perm_r;
      m_perm_cInv = rhs.m_perm_cInv;
    }

    void _setMatAttr() {
      // Specify Lower|Upper fill mode.
      cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
      CHECK_CUSPARSE( cusparseSpMatSetAttribute(m_L.descriptor(), CUSPARSE_SPMAT_FILL_MODE,
                                                &fillmode, sizeof(fillmode)) );
      fillmode = CUSPARSE_FILL_MODE_UPPER;
      CHECK_CUSPARSE( cusparseSpMatSetAttribute(m_U.descriptor(), CUSPARSE_SPMAT_FILL_MODE,
                                                &fillmode, sizeof(fillmode)) );

      // Specify Unit|Non-Unit diagonal type.
      cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
      CHECK_CUSPARSE( cusparseSpMatSetAttribute(m_L.descriptor(), CUSPARSE_SPMAT_DIAG_TYPE,
                                                &diagtype, sizeof(diagtype)) );
      CHECK_CUSPARSE( cusparseSpMatSetAttribute(m_U.descriptor(), CUSPARSE_SPMAT_DIAG_TYPE,
                                                &diagtype, sizeof(diagtype)) );
    }

};
} // namespace GPU
