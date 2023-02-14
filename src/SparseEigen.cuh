#pragma once

namespace GPU {
#define ASSERT(condition, error, ...)                                           \
{                                                                               \
  if (!(condition)) {                                                           \
    fprintf(stderr, "(%s: %d) %s: ", __FILE__, __LINE__, __func__);             \
    fprintf(stderr, error, __VA_ARGS__);                                        \
    fprintf(stderr, "\n");                                                      \
    exit(1);                                                                    \
  }                                                                             \
}

enum EigshResultKind {
  LM = 0,
  LA = 1,
  SM = 2,
  SA = 3
};

template<typename Scalar>
std::vector<size_t> argsort(Eigen::Vector<Scalar, Eigen::Dynamic> &v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::sort(idx.begin(), idx.end(), [&v](size_t a, size_t b){ return v[a] < v[b]; });
  
  return idx;
}

template <typename Scalar>
class Eigsh {
  public:
    using RealType = typename real_type<Scalar>::RealType;
    using CudaType = typename real_type<Scalar>::CudaType;
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
    using VectorR = Eigen::Vector<RealType, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Eigsh(cusparseLinearOperator<Scalar> &a)
      : A(a), uu(0), v(a.rows()), vv(a.rows()), one(1), zero(0), mone(-1)
    {
      CHECK_CUSPARSE( cusparseCreate(&m_cusparseHandle) );
      CHECK_CUBLAS( cublasCreate(&m_cublasHandle) );
    }

    ~Eigsh() {
      CHECK_CUSPARSE( cusparseDestroy(m_cusparseHandle) );
      CHECK_CUBLAS( cublasDestroy(&m_cublasHandle) );
    }

    // Prohibit copy and assignment
    Eigsh(const Eigsh&) = delete;
    Eigsh& operator=(const Eigsh&) = delete;

    void solve(int k, EigshResultKind which = LM,
              int ncv = 0, size_t maxiter = 0, double tol = 0,
              bool return_eigenvectors = true)
    {
      using std::min;
      using std::max;

      int n = A.rows();
      ASSERT(A.rows() == A.cols(), "input must be square matrix (shape: %d, %d)", A.rows(), A.cols());
      ASSERT(k > 0, "k must be greater than 0 (actual: %d)", k);
      ASSERT(k < n, "k must be smaller then n (actual: %d)", k);

      if (ncv <= 0) {
        ncv = min(max(2 * k, k + 32), n - 1);
      } else {
        ncv = min(max(ncv, k + 2), n - 1);
      }
      if (maxiter <= 0) {
        maxiter = 10 * n;
      }
      if (tol <= 0) {
        tol = std::numeric_limits<RealType>::epsilon();
      }

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      int ccMajor = deviceProp.major;

      // Initiate data structures for lanczos iteration
      // Main diagonal
      cuVector<Scalar> alpha(ncv);
      Vector alphaHost(ncv);
      // Sub-diagonals. Althogh beta is always real, it is convinient to have same types as other vectors.
      cuVector<Scalar> beta(ncv);
      Vector betaHost(ncv);
      cuVector<Scalar> beta_k(k);
      Vector beta_kHost(k);
      // Lanczos vectors. Orthonormal.
      cuMatrix<Scalar> V(n, ncv);
      alpha.setZero();
      beta.setZero();

      // Set initial lanczos vector
      cuVector<Scalar> u(Eigen::Vector<Scalar, Eigen::Dynamic>::Random(n));    
      u.norm(m_cublasHandle, (RealType *) beta[0]);                     // beta[0] is just used as a temp buffer
      u.divideByReal((RealType *) beta[0], V.col(0));   // V.col(0) = u / beta[0]

      uu.resize(ncv);

      // Lanczos iteration
      _lanczos(V, u, alpha, beta, 0, ncv);

      size_t iter = ncv;
      // Solve for eigen-pairs of the tridiagonal matrix
      alphaHost = alpha.get();
      betaHost = beta.get();
      cuMatrix<Scalar> s(ncv, k);
      cuVector<Scalar> w(k);
      _eigsh_solve_ritz(alphaHost, betaHost, beta_kHost, k, which, w, s, true);
      
      cuMatrix<Scalar> x(n, k);
      // x = V * s
      CHECK_CUBLAS( cublasSetPointerMode(m_cublasHandle, CUBLAS_POINTER_MODE_DEVICE) );
      CHECK_CUBLAS( _gemm(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, k, ncv,
                          one.data(), V.data(), n,
                          s.data(), ncv,
                          zero.data(), x.data(), n,
                          ccMajor) );

      // Compute residual
      // beta_k = beta[-1] * s[-1, :]
      CHECK_CUBLAS( _copy(m_cublasHandle, k, s.data() + s.rows() - 1, s.rows(), beta_k.data(), 1) );
      CHECK_CUBLAS( _scal(m_cublasHandle, k, (RealType *) beta[beta.size() - 1], beta_k.data(), 1) );

      RealType res = beta_k.norm(m_cublasHandle);

      while (res > tol && iter < maxiter) {
        // Setup for thick-restart
        CHECK_CUDA( cudaMemset(beta.data(), 0, k * sizeof(Scalar)) ); // beta[:k] = 0;
        CHECK_CUDA( cudaMemcpy(alpha.data(), w.data(), k * sizeof(Scalar), cudaMemcpyDeviceToDevice) ); // alpha[:k] = w
        CHECK_CUDA( cudaMemcpy(V.data(), x.data(), n * k * sizeof(Scalar), cudaMemcpyDeviceToDevice) ); // V[:, :k] = x

        // Orthogonalize: u -= V[:, :k] * (V[:, :k].H * u)
        CHECK_CUBLAS( cublasSetPointerMode(m_cublasHandle, CUBLAS_POINTER_MODE_DEVICE) );
        CHECK_CUBLAS( _gemv(m_cublasHandle, CUBLAS_OP_HERMITAN,
                            n, k,
                            one.data(),
                            V.data(), n,
                            u.data(), 1,
                            zero.data(), uu.data(), 1) );
        CHECK_CUBLAS( _gemv(m_cublasHandle, CUBLAS_OP_N,
                            n, k,
                            mone.data(),
                            V.data(), n,
                            uu.data(), 1,
                            one.data(), u.data(), 1) );
        u.norm(m_cublasHandle, (RealType *) uu[0]);     // uu[0] is used as a temp buffer
        u.divideByReal((RealType *) uu[0], V.col(k));   // V.col(k) = u / uu[0] = u / ||u||
        
        v = V.col(k);
        A.matvec(m_cusparseHandle, v, u);
        v.dotc(m_cublasHandle, u.data(), alpha[k]);
        // u = u - alpha[k] * v - V[:, :k] * beta_k
        vv.setZero();
        vv.axpy(m_cublasHandle, v.data(), alpha[k]);
        CHECK_CUBLAS( _gemv(m_cublasHandle, CUBLAS_OP_N,
                            n, k,
                            one.data(),
                            V.data(), n,
                            beta_k.data(), 1,
                            one.data(), vv.data(), 1) );
        u.axpy(m_cublasHandle, vv.data(), mone.data());

        // Lanczos iteration
        _lanczos(V, u, alpha, beta, k + 1, ncv);

        iter += ncv - k;
        // Solve for eigen-pairs of the tridiagonal matrix
        alphaHost = alpha.get();
        betaHost = beta.get();
        _eigsh_solve_ritz(alphaHost, betaHost, beta_kHost, k, which, w, s);
        // x = V * s
        CHECK_CUBLAS( _gemm(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, k, ncv,
                            one.data(), V.data(), n,
                            s.data(), ncv,
                            zero.data(), x.data(), n,
                            ccMajor) );
        
        // Compute residual
        // beta_k = beta[-1] * s[-1, :]
        CHECK_CUBLAS( _copy(m_cublasHandle, k, s.data() + s.rows() - 1, s.rows(), beta_k.data(), 1) );
        CHECK_CUBLAS( _scal(m_cublasHandle, k, (RealType *) beta[beta.size() - 1], beta_k.data(), 1) );
        res = beta_k.norm(m_cublasHandle);
      }
      m_eigenvalues = w.get();
      m_eigenvectors = x.get();
      m_sortIdx = argsort(m_eigenvalues);
    }

    VectorR eigenvalues() const {
      return m_eigenvalues(m_sortIdx);
    }

    VectorR eigenvectors() const {
      return m_eigenvectors(Eigen::placeholders::all, m_sortIdx);
    }

  private:
    cublasHandle_t m_cublasHandle;
    cusparseHandle_t m_cusparseHandle;

    cusparseLinearOperator<Scalar> &A;

    // Neat constants in device memory
    cuScalar<Scalar> one;   // 1
    cuScalar<Scalar> zero;  // 0
    cuScalar<Scalar> mone;  // -1

    // Scratch pads for intermediate calculation results
    cuVector<Scalar> uu;
    cuVector<Scalar> v;
    cuVector<Scalar> vv;

    // Results
    VectorR m_eigenvalues;
    Matrix m_eigenvectors;
    std::vector<size_t> m_sortIdx;

    void _lanczos(cuMatrix<Scalar> &V, cuVector<Scalar> &u,
                  cuVector<Scalar> &alpha, cuVector<Scalar> &beta,
                  int start, int end) 
    {
      int n = V.rows();
      v = V.col(start);
      for (int i = start; i < end; ++i) {
        A.matvec(m_cusparseHandle, v, u);             // u = A * v

        // Call dotc: alpha[i] = v dot u
        v.dotc(m_cublasHandle, u.data(), alpha[i]);

        // Orthogonalize: u = u - alpha[i] * v - beta[i - 1] * V[i - 1]
        vv.setZero();
        if (i > 0) {
          vv.axpy(m_cublasHandle, V.col(i - 1), beta[i - 1]);
        }
        vv.axpy(m_cublasHandle, v.data(), alpha[i]);
        u.axpy(m_cublasHandle, vv.data(), mone.data());

        // Reorthogonalize: u -= V * (V.H * u)
        CHECK_CUBLAS( cublasSetPointerMode(m_cublasHandle, CUBLAS_POINTER_MODE_DEVICE) );
        CHECK_CUBLAS( _gemv(m_cublasHandle, CUBLAS_OP_HERMITAN,
                            n, i + 1,
                            one.data(),
                            V.data(), n,
                            u.data(), 1,
                            zero.data(), uu.data(), 1) );
        CHECK_CUBLAS( _gemv(m_cublasHandle, CUBLAS_OP_N,
                            n, i + 1,
                            mone.data(),
                            V.data(), n,
                            uu.data(), 1,
                            one.data(), u.data(), 1) );
        addInPlace<CudaType>(alpha[i], uu[i]);

        // Call nrm2: beta[i] = ||u||
        u.norm(m_cublasHandle, (RealType *) beta[i]);

        // Break here as the normalization below touches V.col(i + 1)
        if (i >= end - 1) {
          break;
        }

        // Normalize
        eigshNormalize<CudaType>(V.col(i + 1), v.data(), n, u.data(), beta[i]);
      }
    }

    void _eigsh_solve_ritz(Vector &alpha, Vector &beta, Vector &beta_k, int k, EigshResultKind which,
                           cuVector<Scalar> &w, cuMatrix<Scalar> &s, bool firstRun = false) {
      Matrix t(alpha.rows(), alpha.rows());

      // t is Hermitian. Only the lower triangular part is referenced by eigen solver 
      t.diagonal() = alpha;
      t.diagnoal(-1) = beta.head(beta.rows() - 1);
      if (!firstRun) {
        t.block(k, 0, 1, k) = beta_k.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Matrix> eigh(t);

      // Pick up k ritz-values and ritz-vectors
      // Results returned by the solver are already sorted in increasing order.
      if (which == LA) {
        w = eigh.eigenvalues().tail(k);
        s = eigh.eigenvectors().rightCols(k);
      } else if (which == SA) {
        w = eigh.eigenvalues().head(k);
        s = eigh.eigenvectors().leftCols(k);
      } else if (which == LM) {
        VectorR wAbs = eigh.eigenvalues().cwiseAbs();
        std::vector<size_t> idx = argsort(wAbs);
        w = eigh.eigenvalues()(idx).tail(k);
        s = eigh.eigenvectors()(Eigen::placeholders::all, idx).rightCols(k);
      } else if (which == SM) {
        VectorR wAbs = eigh.eigenvalues().cwiseAbs();
        std::vector<size_t> idx = argsort(wAbs);
        w = eigh.eigenvalues()(idx).head(k);
        s = eigh.eigenvectors()(Eigen::placeholders::all, idx).leftCols(k);
      }
    }
};
}
