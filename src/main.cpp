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
  int N = 60;

  std::cout << "Max number of threads allowed: " << Eigen::nbThreads() << "." << std::endl;
  using precision = double;

  SparseMatrix<std::complex<precision>, Eigen::ColMajor> H;
  std::vector<std::complex<precision>> S;

  for (int i = 0; i < N * N; ++i) {
    S.emplace_back(1, 0);
  }
  
  GetHamiltonian(N, S, 0.06, H);

  std::cout << H.rows() << std::endl;
  std::cout << H.cols() << std::endl;
  std::cout << H.nonZeros() << std::endl;

  // cout << Matrix<std::complex<float>, -1, -1>(H) << endl << endl;
  // Vector<std::complex<float>, -1> d(H.rows());
  // d.setConstant(std::complex<float>(0.5));
  // H.diagonal() -= d;
  // cout << Matrix<std::complex<float>, -1, -1>(H) << endl;

  
  // SparseLU<SparseMatrix<std::complex<float>>, Eigen::COLAMDOrdering<int>> lu;
  // {
  //   Timer timer("LU decomp");
  //   lu.isSymmetric(true);
  //   lu.analyzePattern(H);
  //   lu.factorize(H);
  // }

  // GPU::cusparseLU<complex<float>> luG(lu);
  GPU::cuparseCsrMatrix<complex<precision>> Hdevice(H);
  GPU::Eigsh<complex<precision>> eigsh(Hdevice);
  Vector<precision, -1> E;
  Matrix<complex<precision>, -1, -1> V;
  {
    Timer timer("Eigen Solve");
    eigsh.solve(10, GPU::LM);
    E = eigsh.eigenvalues();
    V = eigsh.eigenvectors();
  }

  cout << "Result:" << endl;
  cout << E.transpose() << endl;

  // std::vector<unsigned long> shape{(unsigned long) L.nonZeros()};
  // std::vector<unsigned long> shapeCol{(unsigned long) L.cols() + 1};
  // npy::SaveArrayAsNumpy("rowidx.npy", false, shape.size(), shape.data(), L.innerIndexPtr());
  // npy::SaveArrayAsNumpy("data.npy", false, shape.size(), shape.data(), L.valuePtr());
  // npy::SaveArrayAsNumpy("colptr.npy", false, shapeCol.size(), shapeCol.data(), L.outerIndexPtr());

  return 0;
}