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
class cusparseCsrMatrix : public cusparseLinearOperator<Scalar> {
  public:
    using CudaType = typename real_type<Scalar>::CudaType;
    cusparseCsrMatrix() {}

    // Construct cusparseCsrMatrix from a generic Eigen sparse matrix. The input must be compressed.
    cusparseCsrMatrix(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &matrix) {
      setMatrix(matrix);
    }

    // Construct cusparseCsrMatrix from a generic Eigen sparse matrix. The input must be compressed.
    cusparseCsrMatrix(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &matrix) {
      setMatrix(matrix);
    }

    /*
     *cusparseCsrMatrix(cudaStream_t stream, const Eigen::SparseMatrix<Scalar> &matrix) {
     *m_stream = stream;
     *setMatrix(matrix);
     *}
    */

    // Construct cusparseCsrMatrix directly from c arrays.
    cusparseCsrMatrix(const void *data, const void *rowPtr, const void *colIdx, const int nnz, const int nrow, const int ncol) {
      m_nnz = nnz;
      m_rows = nrow;
      m_cols = ncol;

      _setMatrix(rowPtr, colIdx, data, cudaMemcpyHostToDevice);
    }

    // Copy constructor
    cusparseCsrMatrix(const cusparseCsrMatrix<Scalar> &rhs) {
      _copy(rhs);
    }

    // Destructor
    ~cusparseCsrMatrix() {
      _destroy();
    }

    // Copy assigment operator
    cusparseCsrMatrix<Scalar>& operator=(const cusparseCsrMatrix<Scalar> &rhs) {
      if (this != &rhs) {
        _copy(rhs);
      }
      return *this;
    }

    cusparseCsrMatrix<Scalar>& operator=(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &rhs) {
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
      CHECK_CUSPARSE( cusparseCreateDnVec(&descrOut, out.size(), out.data(), CUDA_DTYPE<Scalar>) );
      CHECK_CUSPARSE( cusparseCreateDnVec(&descrRhs, rhs.size(), rhs.data(), CUDA_DTYPE<Scalar>) );

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
                                CUDA_DTYPE<Scalar>, CUSPARSE_SPMV_CSR_ALG1,
                                &bufSz) );
        CHECK_CUDA( cudaMalloc((void **) &m_workBuf, (size_t) (1.2 * bufSz)) );
      }
      
      // Calculate
      CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, m_descr, descrRhs, &beta, descrOut,
                                   CUDA_DTYPE<Scalar>, CUSPARSE_SPMV_CSR_ALG1,
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
    int *rowPosPtr() const { return m_dCsrRowPtr; }

    /* Pointer to column indices array.
     *Note: the pointer points to an address in GPU memory.
    */ 
    int *colIdxPtr() const { return m_dCsrColIdx; }

    /* Pointer to non-zero element array.
     *Note: the pointer points to an address in GPU memory.
    */ 
    CudaType *valuePtr() const { return m_dCsrVal; }

    // The cusparse sparse matrix descriptor of this matrix.
    cusparseSpMatDescr_t descriptor() const { return m_descr; }

  private:
    cudaStream_t m_stream = NULL;
    cusparseSpMatDescr_t m_descr = NULL;
    int *m_dCsrRowPtr = nullptr;
    int *m_dCsrColIdx = nullptr;
    CudaType *m_dCsrVal = nullptr;
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

    void _copy(const cusparseCsrMatrix<Scalar> &rhs) {
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
                                        CUDA_DTYPE<Scalar>) );
    }

};

template <typename Scalar>
class cusparseLU : public cusparseLinearOperator<Scalar> {
  public:
    using CudaType = typename real_type<Scalar>::CudaType;
    cusparseLU() {}

    // Construct cusparseLU from solved Eigen SparseLU objects.
    cusparseLU(Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> &lu)
      : m_perm_r(lu.rowsPermutation().indices()), m_perm_cInv(lu.colsPermutation().inverse().eval().indices())//, m_lu(lu)
    {
      Eigen::SparseMatrix<Scalar, Eigen::RowMajor> lcsr, ucsr;
      lu.getCsrLU(lcsr, ucsr);
      m_L = lcsr;
      m_U = ucsr;
      _setMatDescr();
    }

    // Construct cusparseLU from Eigen SparseMatrix
    cusparseLU(Eigen::SparseMatrix<Scalar> &L, Eigen::SparseMatrix<Scalar> &U,
               Eigen::PermutationMatrix<Eigen::Dynamic> perm_r, Eigen::PermutationMatrix<Eigen::Dynamic> perm_c) 
      : m_L(L), m_U(U), m_perm_r(perm_r.indices()), m_perm_cInv(perm_c.inverse().eval().indices())
    {
      _setMatDescr();
    }

    cusparseLU(const void *Ldata, const void *LcolIdx, const void *LrowPtr,
               const void *Udata, const void *UcolIdx, const void *UrowPtr,
               const void *perm_r, const void *perm_cInv,
               const int nnzL, const int nnzU, const int rows)
      : m_L(Ldata, LrowPtr, LcolIdx, nnzL, rows, rows), m_U(Udata, UrowPtr, UcolIdx, nnzU, rows, rows),
        m_perm_r(perm_r, rows, false), m_perm_cInv(perm_cInv, rows, false)
    {
      _setMatDescr();
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
      if (cols() != rhs.size()) {
        fprintf(stderr, "Solve: shape mismatch. Solving system (%d, %d) with rhs (%d)\n", rows(), cols(), rhs.size());
        exit(1);
      }

      out = rhs;
      // cuVector<Scalar> tmp(out.size());

      CHECK_CUSPARSE( cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST) );
      
      // Allocate work buffer and perform analysis if not already allocated
      if (m_dataBuf == nullptr) {
        size_t bufSzL = 0;
        size_t bufSzU = 0;
        
        // Determine buffer size and allocate buffer
        CHECK_CUSPARSE( cusparseCsrsm2_bufferSize(handle, trans, trans,
                                                  rows(), 1, nnzL(), (CudaType *) &alpha, m_descrL,
                                                  m_L.valuePtr(), m_L.rowPosPtr(), m_L.colIdxPtr(),
                                                  rhs.data(), rhs.size(), m_infoL, policy, &bufSzL) )
        CHECK_CUSPARSE( cusparseCsrsm2_bufferSize(handle, trans, trans,
                                                  rows(), 1, nnzU(), (CudaType *) &alpha, m_descrU,
                                                  m_U.valuePtr(), m_U.rowPosPtr(), m_U.colIdxPtr(),
                                                  rhs.data(), rhs.size(), m_infoU, policy, &bufSzU) )
        CHECK_CUDA( cudaMalloc((void **) &m_dataBuf, std::max((size_t) rows(), std::max(bufSzL, bufSzU))) );

        // Perform analysis
        CHECK_CUSPARSE( cusparseCsrsm2_analysis(handle, trans, trans,
                                                rows(), 1, nnzL(), (CudaType *) &alpha, m_descrL,
                                                m_L.valuePtr(), m_L.rowPosPtr(), m_L.colIdxPtr(),
                                                rhs.data(), rhs.size(), m_infoL, policy, m_dataBuf) );
        CHECK_CUSPARSE( cusparseCsrsm2_analysis(handle, trans, trans,
                                                rows(), 1, nnzU(), (CudaType *) &alpha, m_descrU,
                                                m_U.valuePtr(), m_U.rowPosPtr(), m_U.colIdxPtr(),
                                                rhs.data(), rhs.size(), m_infoU, policy, m_dataBuf) );
      }
      
      // Apply row permutation
      // Eigen::Vector<Scalar, -1> ref = rhs.get();  // cpu
      
      // printf("Copy: %d\n", ref.isApprox(out.get()));  // cpu
      out.permute(m_perm_r, m_dataBuf);
      // ref = m_lu.rowsPermutation() * ref;         // cpu
      // printf("Row permutation: %d\n", ref.isApprox(out.get())); // cpu

      // Solve L
      // printf("Solve\n");
      CHECK_CUSPARSE( cusparseCsrsm2_solve(handle, trans, trans,
                                           rows(), 1, nnzL(), (CudaType *) &alpha, m_descrL,
                                           m_L.valuePtr(), m_L.rowPosPtr(), m_L.colIdxPtr(),
                                           out.data(), out.size(), m_infoL, policy, m_dataBuf) );
      // m_lu.matrixL().solveInPlace(ref);  // cpu
      // printf("Solve L: %d\n", ref.isApprox(tmp.get()));  // cpu

      // Solve U
      Scalar one(1);
      CHECK_CUSPARSE( cusparseCsrsm2_solve(handle, trans, trans,
                                           rows(), 1, nnzU(), (CudaType *) &alpha, m_descrU,
                                           m_U.valuePtr(), m_U.rowPosPtr(), m_U.colIdxPtr(),
                                           out.data(), out.size(), m_infoU, policy, m_dataBuf) );
      // m_lu.matrixU().solveInPlace(ref);  // cpu
      // printf("Solve U: %d\n", ref.isApprox(out.get()));  // cpu

      // Apply inverse of column permutation
      out.permute(m_perm_cInv, m_dataBuf);
      // ref = m_lu.colsPermutation().inverse() * ref;  // cpu
      // printf("Col permutation: %d\n", ref.isApprox(out.get()));  // cpu
    }

    // Let Pr * A * Pc.T = L * U. This calculate: out = A^-1 * rhs
    void matvec(cusparseHandle_t handle, const cuVector<Scalar> &rhs, cuVector<Scalar> &out) {
      solve(handle, rhs, m_alpha, out);
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
    cusparseCsrMatrix<Scalar> m_L, m_U;
    cuVector<int> m_perm_r, m_perm_cInv;
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseMatDescr_t m_descrL = NULL;
    cusparseMatDescr_t m_descrU = NULL;
    csrsm2Info_t m_infoL = NULL;
    csrsm2Info_t m_infoU = NULL;
    const Scalar m_alpha = 1;
    // Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> &m_lu;
    void *m_dataBuf = nullptr;

    void _destroy() {
      if (m_dataBuf) {
        CHECK_CUDA( cudaFree(m_dataBuf) );
        m_dataBuf = nullptr;
      }

      if (m_descrU) {
        CHECK_CUSPARSE( cusparseDestroyCsrsm2Info(m_infoL) );
        CHECK_CUSPARSE( cusparseDestroyCsrsm2Info(m_infoU) );
        CHECK_CUSPARSE( cusparseDestroyMatDescr(m_descrL) );
        CHECK_CUSPARSE( cusparseDestroyMatDescr(m_descrU) );
        m_infoL = NULL;
        m_infoU = NULL;
        m_descrL = NULL;
        m_descrU = NULL;
      }
    }

    void _copy(const cusparseLU<Scalar> &rhs) {
      _destroy();
      m_L = rhs.m_L;
      m_U = rhs.m_U;
      m_perm_r = rhs.m_perm_r;
      m_perm_cInv = rhs.m_perm_cInv;
      _setMatDescr();
    }

    void _setMatDescr() {
      // Create matrix descriptors
      CHECK_CUSPARSE(cusparseCreateMatDescr(&m_descrU));
      CHECK_CUSPARSE(cusparseCreateMatDescr(&m_descrL));
      // Specify Index Base
      CHECK_CUSPARSE(cusparseSetMatIndexBase(m_descrL, CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(m_descrU, CUSPARSE_INDEX_BASE_ZERO));
      // Specify Lower|Upper fill mode.
      CHECK_CUSPARSE(cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER));
      CHECK_CUSPARSE(cusparseSetMatFillMode(m_descrU, CUSPARSE_FILL_MODE_UPPER));
      // Specify Unit|Non-Unit diagonal type.
      CHECK_CUSPARSE(cusparseSetMatDiagType(m_descrL, CUSPARSE_DIAG_TYPE_UNIT));
      CHECK_CUSPARSE(cusparseSetMatDiagType(m_descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

      // Create Info
      CHECK_CUSPARSE(cusparseCreateCsrsm2Info(&m_infoL));
      CHECK_CUSPARSE(cusparseCreateCsrsm2Info(&m_infoU));
    }

};
} // namespace GPU

/* Load a LU directory to a cusparseLU object. */
template <typename Scalar, typename RealType>
void loadLU(GPU::cusparseLU<Scalar> &luG, RealType sigma) {
  namespace fs = std::filesystem;
  std::string dirname = "lu-" + std::to_string(sigma);
  if (!fs::exists(dirname)) {
    fprintf(stderr, "Directory %s doesn't exist.\n", dirname.c_str());
    exit(1);
  }

  std::string fnLdata = dirname + "/Ldata.npy";
  std::string fnLrowptr = dirname + "/Lrptr.npy";
  std::string fnLcolidx = dirname + "/Lcidx.npy";
  std::string fnUdata = dirname + "/Udata.npy";
  std::string fnUrowptr = dirname + "/Urptr.npy";
  std::string fnUcolidx = dirname + "/Ucidx.npy";
  std::string fnPermR = dirname + "/perm_r.npy";
  std::string fnPermCI = dirname + "/perm_cI.npy";

  std::vector<Scalar> Ldata, Udata;
  std::vector<int> Lrptr, Urptr, Lcidx, Ucidx, perm_r, perm_cI;
  std::vector<unsigned long> shape;
  bool fortran_order;

  npy::LoadArrayFromNumpy(fnLdata, shape, fortran_order, Ldata);
  npy::LoadArrayFromNumpy(fnLrowptr, shape, fortran_order, Lrptr);
  npy::LoadArrayFromNumpy(fnLcolidx, shape, fortran_order, Lcidx);

  npy::LoadArrayFromNumpy(fnUdata, shape, fortran_order, Udata);
  npy::LoadArrayFromNumpy(fnUrowptr, shape, fortran_order, Urptr);
  npy::LoadArrayFromNumpy(fnUcolidx, shape, fortran_order, Ucidx);

  npy::LoadArrayFromNumpy(fnPermR, shape, fortran_order, perm_r);
  npy::LoadArrayFromNumpy(fnPermCI, shape, fortran_order, perm_cI);

  luG = GPU::cusparseLU<Scalar>(Ldata.data(), Lcidx.data(), Lrptr.data(),
                                Udata.data(), Ucidx.data(), Urptr.data(),
                                perm_r.data(), perm_cI.data(),
                                Ldata.size(), Udata.size(), Lrptr.size() - 1);
}
