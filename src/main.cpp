#include "InternalInclude.h"

int main() {
  using Eigen::Matrix;
  using Eigen::Vector;
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  int N = 100;

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

  SparseMatrix<std::complex<float>> L, U;
  lu.getCscLU(L, U);

  std::cout << L.nonZeros() << std::endl;

  std::vector<unsigned long> shape{(unsigned long) L.nonZeros()};
  std::vector<unsigned long> shapeCol{(unsigned long) L.cols() + 1};
  npy::SaveArrayAsNumpy("rowidx.npy", false, shape.size(), shape.data(), L.innerIndexPtr());
  npy::SaveArrayAsNumpy("data.npy", false, shape.size(), shape.data(), L.valuePtr());
  npy::SaveArrayAsNumpy("colptr.npy", false, shapeCol.size(), shapeCol.data(), L.outerIndexPtr());

  return 0;
}