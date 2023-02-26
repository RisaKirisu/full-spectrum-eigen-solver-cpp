#pragma once

template <typename T>
inline void setSqrDiag(Eigen::Matrix<T, -1, -1> &A, const int k, T val) {
  if (std::abs(k) >= A.rows()) {
    fprintf(stderr, "Cannot set %d-diagonal on a square matrix of size %ld.\n", k, A.rows());
    exit(1);
  }

  int i = std::max(0, -k);
  int j = std::max(0, k);
  while(i < A.rows() && j < A.rows()) {
    A(i++, j++) = val;
  }
}

template <typename T>
inline Eigen::SparseMatrix<T> kron3(const Eigen::SparseMatrix<T> &A, const Eigen::SparseMatrix<T> &B, const Eigen::SparseMatrix<T> &C) {
  return Eigen::kroneckerProduct(A, Eigen::kroneckerProduct(B, C));
}

template <typename T>
inline void hstack(const Eigen::SparseMatrix<T> &l, const Eigen::SparseMatrix<T> &r, Eigen::SparseMatrix<T, Eigen::ColMajor> &output) {
  if (l.rows() != r.rows()) {
    fprintf(stderr, "Cannot hstack matrices of different heights: (%ld, %ld) and (%ld, %ld).\n",
            l.rows(), l.cols(), r.rows(), r.cols());
    exit(1);
  }
  output.resize(l.rows(), l.cols() + r.cols());
  output.leftCols(l.cols()) = l;
  output.rightCols(r.cols()) = r;
}

template <typename T>
inline void vstack(const Eigen::SparseMatrix<T> &l, const Eigen::SparseMatrix<T> &r, Eigen::SparseMatrix<T, Eigen::RowMajor> &output) {
  if (l.cols() != r.cols()) {
    fprintf(stderr, "Cannot vstack matrices of different widths: (%ld, %ld) and (%ld, %ld).\n",
            l.rows(), l.cols(), r.rows(), r.cols());
    exit(1);
  }
  output.resize(l.rows() + r.rows(), l.cols());
  output.topRows(l.rows()) = l;
  output.bottomRows(r.rows()) = r;
}


template <typename T>
int GetHamiltonian(const int N, const std::vector<std::complex<T>> &Sdiag, const double delta_o, Eigen::SparseMatrix<std::complex<T>> &result) {
  typedef std::complex<T> cT;
  using Eigen::Matrix;
  using Eigen::SparseMatrix;

  Matrix<cT, 2, 2> sz {{1, 0}, {0, -1}};
  Matrix<cT, 2, 2> sx {{0, 1}, {1, 0}};
  Matrix<cT, 2, 2> sy {{0, cT(0, 1)}, {cT(0, -1), 0}};

  Matrix<cT, -1, -1> t0(N, N);
  t0.setZero();
  setSqrDiag<cT>(t0, 0, 1);

  Matrix<cT,-1, -1> t1(N, N);
  t1.setZero();
  setSqrDiag<cT>(t1, 1, 1);
  setSqrDiag<cT>(t1, -1, 1);
  t1(0, N-1) = 1;
  t1(N-1, 0) = 1;

  Matrix<cT, -1, -1> tt1(N, N);
  tt1.setZero();
  setSqrDiag<cT>(tt1, 2, 1);
  setSqrDiag<cT>(tt1, -2, 1);
  tt1(0, N-2) = 1;
  tt1(N-2, 0) = 1;
  tt1(1, N-1) = 1;
  tt1(N-1, 1) = 1;

  const T t = 0.25;
  const T tt = -0.031863;
  const T ttt = 0.016487;
  const T tttt = 0.0076112;
  const T mu = -0.16235;

  SparseMatrix<cT> ssz = sz.sparseView();
  SparseMatrix<cT> ssx = sx.sparseView();
  SparseMatrix<cT> ssy = sy.sparseView();
  SparseMatrix<cT> st0 = t0.sparseView();
  SparseMatrix<cT> st1 = t1.sparseView();
  SparseMatrix<cT> stt1 = tt1.sparseView();


  result = -mu * kron3(ssz, st0, st0)
           - t * (kron3(ssz, st1, st0) + kron3(ssz, st0, st1))
           - tt * kron3(ssz, st1, st1)
           - ttt * (kron3(ssz, stt1, st0) + kron3(ssz, st0, stt1))
           - tttt * kron3(ssz, stt1, stt1);

  SparseMatrix<cT> dform = 0.5 * kroneckerProduct(st1, st0) - kroneckerProduct(st0, st1);

  SparseMatrix<cT> S(N * N, N * N);
  for (int i = 0; i < N * N; ++i) {
    S.insert(i, i) = Sdiag[i];
  }

  S *= delta_o;
  S = S * dform + dform * S;

  SparseMatrix<cT, Eigen::ColMajor> topCol;
  SparseMatrix<cT, Eigen::ColMajor> botCol;
  SparseMatrix<cT, Eigen::RowMajor> Srow;
  SparseMatrix<cT> zeros(N * N, N*N);
  SparseMatrix<cT> Sadj = S.adjoint();
  hstack(zeros, S, topCol);
  hstack(Sadj, zeros, botCol);
  vstack(topCol, botCol, Srow);
  S = Srow;

  result += S;
  cT zero(0);
  result.prune(zero);
  result.makeCompressed();
  
  return 0;
}