#include "InternalInclude.h"

#ifdef USE_DOUBLE
typedef std::complex<double> Scalar;
typedef double RealType;
#else
typedef std::complex<float> Scalar;
typedef float RealType;
#endif

int main(int argc, char *argv[]) {
  using Eigen::SparseMatrix;
  using std::string;

#ifdef USE_DOUBLE
  printf("Using double precision.\n");
#else
  printf("Using single precision.\n");
#endif

  // Make output unbuffered
  setbuf(stdout,NULL);
  setbuf(stderr,NULL);
  
  int N;
  string fnS_r;
  string fnS_i;
  string fnInterval;
  int k;
  double delta_o;
  int nthreads;

  // Get commandline arguments.
  if (argc != 7) {
    printFactorizeUsage();
  }
  N = std::stoi(argv[1]);
  delta_o = std::stod(argv[2]);
  fnS_r = string(argv[3]);
  fnS_i = string(argv[4]);
  fnInterval = string(argv[5]);
  nthreads = std::stoi(argv[6]);
  printf("Args: %d, %f, %s, %s, %s, %d\n", N, delta_o, fnS_r.c_str(), fnS_i.c_str(), fnInterval.c_str(), nthreads);

  // Collect system info
  nthreads = (nthreads == -1) ? std::thread::hardware_concurrency() - 2 : nthreads;
  nthreads = std::max((int) (nthreads / 2.5), 1);   // Eigen's Sparse LU decomposition routine also spawn threads
#ifdef USE_DOUBLE
  nthreads /= 2;
#endif

  // Read S and intervals from provided files.
  int Nsq = N * N;
  std::vector<RealType> S_r, S_i, interval;
  loadFromFile(fnS_r, S_r, Nsq);
  loadFromFile(fnS_i, S_i, Nsq);
  loadFromFile(fnInterval, interval);

  std::vector<Scalar> S;
  for (int i = 0; i < Nsq; ++i) {
    S.emplace_back(S_r[i], S_i[i]);
  }

  // Construct H.
  SparseMatrix<Scalar, Eigen::ColMajor> H;
  GetHamiltonian(N, S, 0.06, H);

  printCurrentTime();
  printf(": Finished constructing hamiltonian of size %d * %d.\n", N, N);
  Eigen::saveMarket(H, "H.mm");

  return 0;
}