#include "InternalIncludeCuda.h"
// #include "InternalInclude.h"


int main() {
  using Eigen::Matrix;
  using Eigen::Vector;
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using std::cout;
  using std::cin;
  using std::endl;
  using std::complex;
  int N = 400;

  std::cout << "Max number of threads allowed: " << Eigen::nbThreads() << "." << std::endl;

  SparseMatrix<std::complex<float>, Eigen::ColMajor> H;
  std::vector<std::complex<float>> S;

  for (int i = 0; i < N * N; ++i) {
    S.emplace_back(1, 0);
  }
  
  GetHamiltonian(N, S, 0.06, H);

  std::cout << H.rows() << std::endl;
  std::cout << H.cols() << std::endl;
  std::cout << H.nonZeros() << std::endl;
  
  SparseLU<SparseMatrix<std::complex<float>>, Eigen::COLAMDOrdering<int>> lu;
  {
    Timer timer("LU decomp");
    lu.isSymmetric(true);
    lu.analyzePattern(H);
    lu.factorize(H);
  }

  GPU::cusparseLU<complex<float>> luG(lu);

  GPU::Eigsh<complex<float>> eig(luG);
  eig.solve(10);

  // cout << "lu nnzL: " << lu.nnzL() << endl;
  // cout << "lu nnzU: " << lu.nnzU() << endl;
  
  // std::srand((unsigned int) time(0));
  // Matrix<std::complex<float>, -1, -1> b = Matrix<std::complex<float>, -1, -1>::Random(2 * N * N, N);
  // Matrix<std::complex<float>, -1, -1> refX(b.rows(), b.cols());
  // Matrix<std::complex<float>, -1, -1> x1(b.rows(), b.cols());
  // Matrix<std::complex<float>, -1, -1> x2(b.rows(), b.cols());

  // refX = lu.solve(b);
  // cout << "Press enter to continue... ";
  // cin.ignore();

  // {
  //   Timer timer("toSparse");
  //   SparseMatrix<std::complex<float>, Eigen::RowMajor> L = lu.matrixL().toSparse();
  //   SparseMatrix<std::complex<float>, Eigen::RowMajor> U = lu.matrixU().toSparse();

    // x1.resize(b.rows(), b.cols());
    // for (int i = 0; i < b.cols(); ++i) {
    //   x1.col(i) = lu.rowsPermutation() * b.col(i);
    // }
    // L.triangularView<Eigen::Lower>().solveInPlace(x1);
    // U.triangularView<Eigen::Upper>().solveInPlace(x1);
    // lu.matrixL().solveInPlace(x1);
    // lu.matrixU().solveInPlace(x1);
    // for (int i = 0; i < b.cols(); ++i) {
    //   x1.col(i) = lu.colsPermutation().inverse() * x1.col(i);
    // }
    
  //   std::cout << "L nnz:" << L.nonZeros() << std::endl;
  //   std::cout << "U nnz:" << U.nonZeros() << std::endl;
  //   cout << "Press enter to continue... ";
  //   cin.ignore();
  // }

  // {
  //   Timer timer("getCsrLU");
  //   SparseMatrix<std::complex<float>, Eigen::RowMajor> L, U;
  //   lu.getCsrLU(L, U);

    // x2.resize(b.rows(), b.cols());
    // for (int i = 0; i < b.cols(); ++i) {
    //   x2.col(i) = lu.rowsPermutation() * b.col(i);
    // }
    // L.triangularView<Eigen::Lower>().solveInPlace(x2);
    // U.triangularView<Eigen::Upper>().solveInPlace(x2);
    // lu.matrixL().solveInPlace(x2);
    // lu.matrixU().solveInPlace(x2);
    // for (int i = 0; i < b.cols(); ++i) {
    //   x2.col(i) = lu.colsPermutation().inverse() * x2.col(i);
    // }
    
  //   std::cout << "L nnz:" << L.nonZeros() << std::endl;
  //   std::cout << "U nnz:" << U.nonZeros() << std::endl;
  //   cout << "Press enter to continue... ";
  //   cin.ignore();
  // }

  // if (refX.isApprox(x1)) {
  //   cout << "PASS" << endl;
  // } else {
  //   cout << "FAIL" << endl;
  // }

  // if (refX.isApprox(x2)) {
  //   cout << "PASS" << endl;
  // } else {
  //   cout << "FAIL" << endl;
  // }

  // if (x1.isApprox(x2)) {
  //   cout << "PASS" << endl;
  // } else {
  //   cout << "FAIL" << endl;
  // }

  // std::vector<unsigned long> shape{(unsigned long) L.nonZeros()};
  // std::vector<unsigned long> shapeCol{(unsigned long) L.cols() + 1};
  // npy::SaveArrayAsNumpy("rowidx.npy", false, shape.size(), shape.data(), L.innerIndexPtr());
  // npy::SaveArrayAsNumpy("data.npy", false, shape.size(), shape.data(), L.valuePtr());
  // npy::SaveArrayAsNumpy("colptr.npy", false, shapeCol.size(), shapeCol.data(), L.outerIndexPtr());

  return 0;
}