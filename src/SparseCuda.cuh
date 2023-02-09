#pragma once


namespace GPU {

template <typename T>
struct cusparse_traits;

template <>
struct cusparse_traits<float> {
  using cuScalar = float;
  cudaDataType dtype = CUDA_R_32F;
};

template <>
struct cusparse_traits<double> {
  using cuScalar = double;
  cudaDataType dtype = CUDA_R_64F;
};

template <>
struct cusparse_traits<std::complex<float>> {
  using cuScalar = cuComplex;
  cudaDataType dtype = CUDA_C_32F;
};

template <>
struct cusparse_traits<std::complex<double>> {
  using cuScalar = cuDoubleComplex;
  cudaDataType dtype = CUDA_C_64F;
};

template <typename Scalar>
class cuVector {
  public:
    cuVector(void *values, int size, bool onDevice) {
      m_size = size;
      cudaMemcpyKind direction = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
      _setVector(values, direction);
    }

    cuVector(const cuVector<Scalar> &rhs) {
      _destroy();
      _copy(rhs);
    }

    ~cuVector() {
      _destroy();
    }

    cuVector& operator= (const cuVector<Scalar> &rhs) {
      if (this == &rhs) {
        _destroy();
        _copy(rhs);
      }
    }

    int size() const { return m_size; }
    void* data() { return m_dValues; }

  private:
    void *m_dValues = nullptr;
    int m_size = 0;

    void _destroy() {
      if (m_dValues) {
        CHECK_CUDA( cudaFree(m_dValues) );
        m_dValues = nullptr;
        m_size = 0;
      }
    }

    void _copy(const cuVector<Scalar> &rhs) {
      m_size = rhs.m_size;
      _setVector(rhs.m_dValues, cudaMemcpyDeviceToDevice);
    }
    void _setVector(void *values, cudaMemcpyKind direction) {
      CHECK_CUDA( cudaMalloc((void**) &m_dValues, size() * sizeof(Scalar)) );
      CHECK_CUDA( cudaMemcpy(m_dValues, values, size() * sizeof(Scalar), direction) );
    }
};

template <typename Scalar>
class cuparseCsrMatrix {
  public:
    cuparseCsrMatrix(Eigen::SparseMatrix<Scalar> &matrix, bool constexp = true) {
      m_constexp = constexp;
      setMatrix(matrix);
    }

    cuparseCsrMatrix(cudaStream_t stream, Eigen::SparseMatrix<Scalar> &matrix, bool constexp = true) {
      m_stream = stream;
      m_constexp = constexp;
      setMatrix(matrix);
    }

    // Copy constructor
    cuparseCsrMatrix(const cuparseCsrMatrix<Scalar> &rhs) {
      _destroy();
      _copy(rhs);
    }

    // Destructor
    ~cuparseCsrMatrix() {
      _destroy();
    }

    // Copy assigment operator
    cuparseCsrMatrix& operator=(const cuparseCsrMatrix<Scalar> &rhs) {
      if (this != &other) {
        _destroy();
        _copy(rhs);
      }
      return *this;
    }


    void setMatrix(Eigen::SparseMatrix<Scalar> &matrix) {
      /* If a cusparse matrix already exist, destroy it and create a new one*/
      _destroy();

      m_nnz = matrix.nonZeros();
      m_rows = matrix.rows();
      m_cols = matrix.cols();

      Eigen::SparseMatrix<Scalar, Eigen::RowMajor> csr(matrix);

      if (!csr.isCompressed()) {
        csr.makeCompressed();
      }

      _setMatrix(csr.outerIndexPtr(), csr.innerIndexPtr(), csr.valuePtr(), cudaMemcpyHostToDevice);
    }

    void matvec(cusparseHandle_t handle, cuVector<Scalar> &rhs, Scalar alpha, Scalar beta, cuVector<Scalar> &out) {
      if (cols() != rhs.size()) {
        fprintf(stderr, "Matvec: shape mismatch. Multiplying (%d, %d) and (%d)\n", rows(), cols(), rhs.size());
        exit(1);
      }

      cusparseDnVecDescr_t descrVec;
      CHECK_CUSPARSE( cusparseCreateConstDnVec(&descrVec, rhs.size(), rhs.data(), cusparse_traits<Scalar>::dtype) );

      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
      // cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_descr, descrVec. &beta, )
      
    }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    int nnz() const { return m_nnz; }

  private:
    cudaStream_t m_stream = NULL;
    cusparseMatDescr_t m_descr = NULL;
    void *m_dCsrRowPtr;
    void *m_dCsrColIdx;
    void *m_dCsrVal;
    int m_nnz = 0;
    int m_rows = 0;
    int m_cols = 0;
    bool m_constexp;

    void _destroy() {
      if (!m_descr) {
        return;
      }
      CHECK_CUSPARSE( cusparseDestroySpMat(m_descr) );
      CHECK_CUDA( cudaFree(m_dCsrRowPtr) );
      CHECK_CUDA( cudaFree(m_dCsrColIdx) );
      CHECK_CUDA( cudaFree(m_dCsrVal) );

      m_descr = NULL;
      m_nnz = 0;
      m_rows = 0;
      m_cols = 0;
    }

    void _copy(const cuparseCsrMatrix &rhs) {
      m_stream = rhs.m_stream;
      m_nnz = rhs.m_nnz;
      m_rows = rhs.m_rows;
      m_cols = rhs.m_cols;
      m_constexp = rhs.m_constexp;

      _setMatrix(rhs.m_dCsrRowPtr, rhs.m_dCsrColIdx, rhs.m_dCsrVal, cudaMemcpyDeviceToDevice);
    }

    void _setMatrix(void *rowPtr, void *colIdx, void *values, cudaMemcpyKind direction) {
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrRowPtr, (rows() + 1) * sizeof(int)) );
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrColIdx, nnz() * sizeof(int)) );
      CHECK_CUDA( cudaMalloc((void **) &m_dCsrVal, nnz() * sizeof(Scalar)) );

      CHECK_CUDA( cudaMemcpy(m_dCsrRowPtr, rowPtr, (rows() + 1) * sizeof(int), direction) );
      CHECK_CUDA( cudaMemcpy(m_dCsrColIdx, colIdx, nnz() * sizeof(int), direction) );
      CHECK_CUDA( cudaMemcpy(m_dCsrVal, values, nnz() * sizeof(Scalar), direction) );

      if (m_constexp) {
        CHECK_CUSPARSE( cusparseCreateConstCsr(&m_descr, rows(), cols(), cnnz(),
                                               m_dCsrRowPtr, m_dCsrColIdx, m_dCsrVal,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               cusparse_traits<Scalar>::dtype) );
      } else {
        CHECK_CUSPARSE( cusparseCreateCsr(&m_descr,rows(), cols(), nnz(),
                                          m_dCsrRowPtr, m_dCsrColIdx, m_dCsrVal,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          cusparse_traits<Scalar>::dtype) );
      }
    }

};

template <typename Scalar>
class SparseLU {
  public:
    SparseLU();

};

}