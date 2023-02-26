#include "InternalIncludeCuda.h"
#include "Utils.h"
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
  using namespace GPU;
  int N = 20;

  std::cout << "Max number of threads allowed: " << Eigen::nbThreads() << "." << std::endl;
  using precision = float;
  using Scalar = complex<precision>;

  SparseMatrix<std::complex<precision>, Eigen::ColMajor> H;
  std::vector<std::complex<precision>> S;

  for (int i = 0; i < N * N; ++i) {
    S.emplace_back(1, 0);
  }
  
  GetHamiltonian(N, S, 0.06, H);

  cusparseHandle_t handle;
  cusparseCreate(&handle);


  // Matrix<Scalar, -1, -1> Hd(H);
  // Matrix<std::complex<double>, -1, -1> Hdf = Hd.cast<std::complex<double>>();

  // cout << Hdf << endl << endl;
  
  // cout << Matrix<std::complex<float>, -1, -1>(H) << endl << endl;
  // Vector<std::complex<float>, -1> d(H.rows());
  // d.setConstant(std::complex<float>(0.5));
  // H.diagonal() -= d;
  // cout << Matrix<std::complex<float>, -1, -1>(H) << endl;

  SparseLU<SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> lu;
  // {
  //   Timer timer("LU decomp");
    lu.isSymmetric(true);
    lu.analyzePattern(H);
    lu.factorize(H);
  // }
  
  saveLU(lu, 0);
  // SparseMatrix<Scalar, Eigen::RowMajor> L, U;
  // lu.getCsrLU(L, U);
  // Vector<int, -1> perm_r = lu.rowsPermutation().indices();
  // Vector<int, -1> perm_cI = lu.colsPermutation().inverse().eval().indices();

  // Vector<float, -1> v1 {{1, 0, 1}};
  // Vector<float, -1> v2 {{0, 1, 0}};
  // Vector<float, -1> v3 {{1, 0, -1}};
  // Matrix<float, -1, -1> H =  4 * v1 * v1.transpose() + 8 * v2 * v2.transpose() + 3 * v3 * v3.transpose();
  // SparseMatrix<float, Eigen::RowMajor> Hs = H.sparseView();

  Vector<Scalar, -1> y = Vector<Scalar, -1>::Random(lu.rows());
  Vector<Scalar, -1> x;
  // {
  //   Timer timer("Eigen");
    x = lu.solve(y);
  // }

  GPU::cusparseLU<Scalar> luG;
  // GPU::cusparseLU<Scalar> luG(lu);
  loadLU(luG, 0);

  GPU::cuVector<Scalar> yG(y);
  GPU::cuVector<Scalar> xG(lu.rows());

  // {
  //   Timer timer("cuSparse first time");
    luG.matvec(handle, yG, xG);
  // }

  // {
  //   Timer timer("cuSparse");
  //   luG.matvec(handle, yG, xG);
  // }
  Vector<Scalar, -1> xx = xG.get();

  // SparseMatrix<Scalar, Eigen::RowMajor> L, U;
  // lu.getCsrLU(L, U);
  // Vector<Scalar, -1> xxx = y;
  // Vector<Scalar, -1> tmp = y;
  // tmp = lu.rowsPermutation() * xxx;
  // L.triangularView<Eigen::Lower>().solveInPlace(tmp);
  // U.triangularView<Eigen::Upper>().solveInPlace(tmp);
  // xxx = lu.colsPermutation().inverse() * tmp;

  Vector<Scalar, -1> ry = H * x;
  Vector<Scalar, -1> ryy = H * xx;
  // Vector<Scalar, -1> ryyy = H * xxx;
  
  printf("%d\n", y.isApprox(ry));
  printf("%e\n", (y - ry).norm());
  printf("%d\n", y.isApprox(ryy));
  printf("%e\n", (y - ryy).norm());
  // printf("%d\n", y.isApprox(ryyy));
  // printf("%e\n", (y - ryyy).norm());
  // GPU::cusparseCsrMatrix<Scalar> Hdevice(H);
  // GPU::Eigsh<Scalar> eigsh(luG);
  // Vector<precision, -1> E;
  // Matrix<Scalar, -1, -1> V;
  // {
  //   printf("Start Solving. \n");
  //   eigsh.solve(300, GPU::LM, 0, 0, 2e-5);
  //   E = eigsh.eigenvalues();
  //   V = eigsh.eigenvectors();
  // }
  // {
  //   Timer timer("Eigen Solve");
  //   printf("Start Solving. \n");
  //   eigsh.solve(300, GPU::LM, 0, 0, 2e-5);
  //   E = eigsh.eigenvalues();
  //   V = eigsh.eigenvectors();
  // }
  // Matrix<Scalar, -1, -1> VtV = V.adjoint() * V;
  // VtV.diagonal().array() -= 1;

  // // cout << "Result:" << endl;
  // // cout << E.transpose().array().inverse() << endl;
  // cout << VtV.norm() << endl;

  // std::vector<unsigned long> shape{(unsigned long) L.nonZeros()};
  // std::vector<unsigned long> shapeCol{(unsigned long) L.cols() + 1};
  // npy::SaveArrayAsNumpy("rowidx.npy", false, shape.size(), shape.data(), L.innerIndexPtr());
  // npy::SaveArrayAsNumpy("data.npy", false, shape.size(), shape.data(), L.valuePtr());
  // npy::SaveArrayAsNumpy("colptr.npy", false, shapeCol.size(), shapeCol.data(), L.outerIndexPtr());
  cusparseDestroy(handle);
  return 0;
}