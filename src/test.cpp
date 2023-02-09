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
  
  std::srand((unsigned int) time(0));
  const int N = 5;

  Matrix<float, N, N> A = Matrix<float, N, N>::Random();
  SparseMatrix<float> spA = A.sparseView();
  SparseLU<SparseMatrix<float>, Eigen::COLAMDOrdering<int>> lu;
  lu.analyzePattern(spA);
  lu.factorize(spA);
  SparseMatrix<float> L, U;
  lu.getCscLU(L, U);

  cout << Matrix<float, N, N>(L) << endl;
  cout << L.nonZeros() << endl;

  cout << SparseMatrix<float, Eigen::RowMajor>(L).isCompressed() << endl;

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