#include "InternalIncludeCuda.h"
// #include "InternalInclude.h"

#ifdef DOUBLE
typedef std::complex<double> Scalar;
typedef double RealType;
#else
typedef std::complex<float> Scalar;
typedef float RealType;
#endif


int main(int argc, char *argv[]) {
  using Eigen::Matrix;
  using Eigen::Vector;
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using std::string;
  using std::cout;
  using std::cin;
  using std::endl;
  using std::complex;
  
  int N;
  string fnS_r;
  string fnS_i;
  string fnInterval;
  double xtol;

  if (argc != 6) {
    printUsage();
  }
  N = std::stoi(argv[1]);
  fnS_r = string(argv[2]);
  fnS_i = string(argv[3]);
  fnInterval = string(argv[4]);
  xtol = std::stod(argv[5]);
  printf("Args: %d, %s, %s, %s, %f\n", N, fnS_r.c_str(), fnS_i.c_str(), fnInterval.c_str(), xtol);

  std::vector<RealType> S_r, S_i;
  std::vector<int> interval;
  readArray<RealType>(fnS_r, N * N, S_r);

  // std::cout << "Max number of threads allowed: " << Eigen::nbThreads() << "." << std::endl;
  // using precision = float;
  // using Scalar = complex<precision>;

  // SparseMatrix<std::complex<precision>, Eigen::ColMajor> H;
  // std::vector<std::complex<precision>> S;

  // for (int i = 0; i < N * N; ++i) {
  //   S.emplace_back(1, 0);
  // }
  
  // GetHamiltonian(N, S, 0.06, H);


  return 0;
}