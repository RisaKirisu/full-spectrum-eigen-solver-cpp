#include "InternalInclude.h"

// int main() {
//   using Eigen::Matrix;
//   using Eigen::Vector;
//   using Eigen::SparseMatrix;
//   using Eigen::SparseLU;
//   using std::cout;
//   using std::endl;
//   int N = 500;
  
//   Matrix<std::complex<float>, -1, -1> A(N, N);
//   for (int i = 0; i < N; ++i) {
//     for (int j = 0; j < N; ++j) {
//       A(i, j) = std::complex<float>(i % 7 + j * 2 % 5 * 1.5, i * 2 + j % 6 * 0.4);
//     }
//   }
  
//   Matrix<std::complex<float>, -1, -1> H = A + A.adjoint();
//   Eigen::SelfAdjointEigenSolver<Matrix<std::complex<float>, -1, -1>> eigsh;
//   {
//     Timer timer("Eigen Solve");
//     for (int i = 0; i < 100; ++i)
//       eigsh.compute(H);
//   }
//   // cout << eigsh.eigenvalues() << endl;
// }

int main() {
  using Eigen::Matrix;
  using Eigen::Vector;
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using std::cout;
  using std::endl;

  Matrix<float, 5, 5> a = Matrix<float, 5, 5>::Random();
  cout << a << endl << endl;
  Matrix<std::complex<float>, 5, 5> b = a;
  cout << b << endl;

  // int n = 100;
  // Matrix<float, -1, -1> t(n, n);
  // t.setZero();
  // Vector<float, -1> d0 = Vector<float, -1>::Random(n);
  // Vector<float, -1> d1 = Vector<float, -1>::Random(n - 1);
  // t.diagonal() = d0;
  // t.diagonal(-1) = d1;

  // {
  //   Timer timer("eigh");
  //   Eigen::SelfAdjointEigenSolver<Matrix<float, -1, -1>> eigh(t);
  //   Vector<float, -1> E = eigh.eigenvalues();
  //   Matrix<float, -1, -1> V = eigh.eigenvectors();
  //   cout << E[0] << endl;
  //   t.diagonal(1) = d1;
  //   Eigen::SelfAdjointEigenSolver<Matrix<float, -1, -1>> eigh2(t);
  //   cout << eigh2.eigenvalues()[0] << endl;
  // }


  // SparseMatrix<float> spA = A.sparseView();
  // SparseLU<SparseMatrix<float>, Eigen::COLAMDOrdering<int>> lu;
  // lu.analyzePattern(spA);
  // lu.factorize(spA);
  // SparseMatrix<float> L, U;
  // Matrix<float, N, N> recA;


  // L = lu.matrixL().toSparse();
  // U = lu.matrixU().toSparse();
  // cout << "L nnz: " << L.nonZeros() << endl;
  // cout << "U nnz: " << U.nonZeros() << endl;
  // recA = L * U;
  // recA = recA * lu.colsPermutation();
  // recA = lu.rowsPermutation().transpose() * recA;
  // // cout << recA << endl;

  // if (!A.isApprox(recA)) {
  //   cout << "FAIL" << endl;
  // } else {
  //   cout << "PASS" << endl;
  // }

  // lu.getCscLU(L, U);
  // cout << "L nnz: " << L.nonZeros() << endl;
  // cout << "U nnz: " << U.nonZeros() << endl;
  // recA = L * U;
  // recA = recA * lu.colsPermutation();
  // recA = lu.rowsPermutation().transpose() * recA;
  // // cout << recA << endl;

  // if (!A.isApprox(recA)) {
  //   cout << "FAIL" << endl;
  // } else {
  //   cout << "PASS" << endl;
  // }



  // for (int i = 0; i < 2; ++i) {
  //   Matrix<float, N, N> A = Matrix<float, N, N>::Random();
  //   SparseMatrix<float> spA = A.sparseView();
  //   SparseLU<SparseMatrix<float>, Eigen::COLAMDOrdering<int>> lu;
  //   lu.analyzePattern(spA);
  //   lu.factorize(spA);
  //   SparseMatrix<float> L, U;
  //   lu.getCscLU(L, U);

  //   Matrix<float, N, N> recA(L * U);
  //   recA = recA * lu.colsPermutation();
  //   recA = lu.rowsPermutation().transpose() * recA;

  //   if (!A.isApprox(recA)) {
  //     cout << "FAIL" << endl;
  //     exit(-1);
  //   }
  // }
  
  // cout << "PASS" << endl;
  // cout << L << endl;
  // cout << U << endl;

  return 0;
}