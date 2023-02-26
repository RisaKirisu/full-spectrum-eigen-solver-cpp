#pragma once

namespace GPU {

template <typename Scalar>
class cuScalar {
  using CudaType = typename real_type<Scalar>::CudaType;
  public:
    cuScalar(Scalar a) : m_a(a) {
      CHECK_CUDA( cudaMalloc((void **) &m_data, sizeof(Scalar)) );
      CHECK_CUDA( cudaMemcpy(m_data, &m_a, sizeof(Scalar), cudaMemcpyHostToDevice) );
    }

    CudaType *data() const {
      return m_data;
    }

    Scalar value() const {
      return m_a;
    }
  private:
    CudaType *m_data;
    Scalar m_a;
};
template <typename Scalar>
class cuVector {
  public:
    using CudaType = typename real_type<Scalar>::CudaType;
    using RealType = typename real_type<Scalar>::RealType;

    cuVector() {}
    // Construct a vector on device using the pointer and size given
    cuVector(const void *values, int size, bool onDevice) {
      m_size = size;
      cudaMemcpyKind direction = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
      _allocate();
      _setVector(values, direction);
    }

    // Construct a vector on device from Eigen vector
    cuVector(const Eigen::Vector<Scalar, Eigen::Dynamic> &vector) {
      m_size = vector.rows();
      _allocate();
      _setVector(vector.data(), cudaMemcpyHostToDevice);
    }

    // Construct an empty vector of given size.
    cuVector(int size) {
      m_size = size;
      _allocate();
    }

    cuVector(const cuVector<Scalar> &rhs) {
      _copy(rhs);
    }

    ~cuVector() {
      _destroy();
    }

    // Make a deep copy of rhs
    cuVector<Scalar>& operator=(const cuVector<Scalar> &rhs) {
      if (this != &rhs) {
        _copy(rhs);
      }
      return *this;
    }

    // Copy size() elements from rhs
    cuVector<Scalar>& operator=(const void *rhs) {
      _setVector(rhs, cudaMemcpyDeviceToDevice);
      return *this;
    }

    cuVector<Scalar>& operator=(const Eigen::Vector<Scalar, Eigen::Dynamic> &rhs) {
      if (rhs.size() != size()) {
        resize(rhs.size());
      }
      _setVector(rhs.data(), cudaMemcpyHostToDevice);
      return *this;
    }


    // cuVector<Scalar>& operator=(const Eigen::Vector<RealType, Eigen::Dynamic> &rhs) {
    //   if (rhs.size() != size()) {
    //     resize(rhs.size());
    //   }
    //   // Convert to complex if Scalar is complex
    //   Eigen::Vector<Scalar, Eigen::Dynamic> tmp = rhs.template cast<Scalar>();
    //   _setVector(tmp.data(), cudaMemcpyHostToDevice);
    //   return *this;
    // }

    // Return a device pointer to indexed element
    CudaType *operator[](size_t index) {
      if (index > size()) {
        fprintf(stderr, "cuVector: Accessing out of bound element.");
        exit(1);
      }
      return m_dValues + index; 
    }

    void resize(int size) {
      _destroy();
      m_size = size;
      _allocate();
    }

    void permute(cuVector<int> &perm, void *buf) {
      _permute(data(), perm.data(), buf, size());
    }

    // Calculate Euclidean norm of the vector and store in out
    void norm(cublasHandle_t handle, RealType *out) const {
      CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
      CHECK_CUBLAS( _nrm2(handle, size(), data(), 1, out) );
    }

    RealType norm(cublasHandle_t handle) const {
      CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) );
      RealType result;
      CHECK_CUBLAS( _nrm2(handle, size(), data(), 1, &result) );
      return result;
    }

    // Devide by rhs and store the result in out. rhs should reside on device memory.
    void divideByReal(const RealType *rhs, CudaType *out) const {
      _divideByReal(data(), rhs, out, size());
    }

    // Calculate dot product with rhs and stor the result in out.
    void dotc(cublasHandle_t handle, const CudaType *rhs, CudaType *out) const {
      CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
      CHECK_CUBLAS( _dotc(handle, size(), data(), 1, rhs, 1, out) );
    }
    
    // Calculate self = alpha * rhs + self
    void axpy(cublasHandle_t handle, const CudaType *rhs, const CudaType *alpha) {
      CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
      _axpy(handle, size(), alpha, rhs, 1, data(), 1);
    }

    // Set all elements to zero.
    void setZero() {
      if (size() == 0) { return; }
      CHECK_CUDA( cudaMemset(data(), 0, size() * sizeof(Scalar)) );
    }
    
    // Get a copy of the vector on host memory.
    Eigen::Vector<Scalar, Eigen::Dynamic> get() const {
      CHECK_CUDA( cudaMemcpy(m_hValues, data(), size() * sizeof(Scalar), cudaMemcpyDeviceToHost) );
      return Eigen::Map<Eigen::Vector<Scalar, Eigen::Dynamic>>(m_hValues, size());
    }

    int size() const { return m_size; }
    CudaType* data() const { return m_dValues; }

  private:
    CudaType *m_dValues = nullptr;
    Scalar *m_hValues = nullptr;
    int m_size = 0;

    void _destroy() {
      if (m_dValues) {
        CHECK_CUDA( cudaFree(m_dValues) );
        CHECK_CUDA( cudaFreeHost(m_hValues) );
        m_dValues = nullptr;
        m_hValues = nullptr;
        m_size = 0;
      }
    }

    void _copy(const cuVector<Scalar> &rhs) {
      // Skip reallocation if rhs has the same size
      if (size() != rhs.size()) {
        resize(rhs.size());
      }
      _setVector(rhs.m_dValues, cudaMemcpyDeviceToDevice);
    }

    void _allocate() {
      CHECK_CUDA( cudaMalloc((void **) &m_dValues, size() * sizeof(Scalar)) );
      CHECK_CUDA( cudaMallocHost((void **) &m_hValues, size() * sizeof(Scalar)) );
    }

    void _setVector(const void *values, cudaMemcpyKind direction) {
      CHECK_CUDA( cudaMemcpy(m_dValues, values, size() * sizeof(Scalar), direction) );
    }
};

// Column major matrix on GPU
template <typename Scalar>
class cuMatrix {
  public:
    using CudaType = typename real_type<Scalar>::CudaType;

    // Note: rhs must be column major
    cuMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix) {
      m_rows = matrix.rows();
      m_cols = matrix.cols();
      _allocate();
      _setMatrix(matrix.data(), cudaMemcpyHostToDevice);
    }

    cuMatrix(int rows, int cols) {
      m_rows = rows;
      m_cols = cols;
      _allocate();
    }
    cuMatrix(Scalar *values, int rows, int cols) {
      m_rows = rows;
      m_cols = cols;
      _allocate();
      _setMatrix(values, cudaMemcpyHostToDevice);
    }

    // Copy constructor
    cuMatrix(const cuMatrix<Scalar> &rhs) {
      _copy(rhs);
    }

    // Assignment operator
    cuMatrix<Scalar>& operator=(const cuMatrix<Scalar> &rhs) {
      if (&rhs != this) {
        _copy(rhs);
      }

      return *this;
    }

    // Note: rhs must be column major
    cuMatrix<Scalar>& operator=(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix) {
      if (rows() != matrix.rows() || cols() != matrix.cols()) {
        resize(matrix.rows(), matrix.cols());
      }
      _setMatrix(matrix.data(), cudaMemcpyHostToDevice);
      return *this;
    }
    
    // Destructor
    ~cuMatrix() {
      _destroy();
    }

    // Get a copy of the matrix on host memory.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> get() {
      CHECK_CUDA( cudaMemcpy(m_hValues, data(), rows() * cols() * sizeof(Scalar), cudaMemcpyDeviceToHost) );
      return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(m_hValues, rows(), cols());
    }

    void resize(int rows, int cols) {
      _destroy();
      m_rows = rows;
      m_cols = cols;
      _allocate();
    }

    CudaType *data() const { return m_dValues; }
    CudaType *col(int index) const { return m_dValues + index * rows(); }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; } 

  private:
    CudaType *m_dValues = nullptr;
    Scalar *m_hValues = nullptr;
    int m_rows = 0;
    int m_cols = 0;

    void _destroy() {
      if (m_dValues) {
        CHECK_CUDA( cudaFree(m_dValues) );
        CHECK_CUDA( cudaFreeHost(m_hValues) );
        m_dValues = nullptr;
        m_hValues = nullptr;
        m_rows = 0;
        m_cols = 0;
      }
    }

    void _copy(const cuMatrix<Scalar> &rhs) {
      // Skip reallocation if rhs has the same size
      if (rows() != rhs.rows() || cols() != rhs.cols()) {
        _destroy();
        m_rows = rhs.m_rows;
        m_cols = rhs.m_cols;
        _allocate();
      }
      _setMatrix(rhs.m_dValues, cudaMemcpyDeviceToDevice);
    }

    void _allocate() {
      CHECK_CUDA( cudaMalloc((void **) &m_dValues, rows() * cols() * sizeof(Scalar)) );
      CHECK_CUDA( cudaMallocHost((void **) &m_hValues, rows() * cols() * sizeof(Scalar)) );
    }

    void _setMatrix(const void *values, cudaMemcpyKind direction) {
      CHECK_CUDA( cudaMemcpy(m_dValues, values, rows() * cols() * sizeof(Scalar), direction) );
    }
};

} // namespace GPU
