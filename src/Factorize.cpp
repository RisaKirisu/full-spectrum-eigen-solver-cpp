#include "InternalInclude.h"


#ifdef USE_DOUBLE
typedef std::complex<double> Scalar;
typedef double RealType;
#else
typedef std::complex<float> Scalar;
typedef float RealType;
#endif

void Factorize(Eigen::SparseMatrix<Scalar, Eigen::ColMajor>  &T,
               std::vector<RealType>                 &intervals,
               int                                     nThreads);

int main(int argc, char *argv[]) {
  using Eigen::SparseMatrix;
  using std::string;

#ifdef USE_DOUBLE
  printf("Using double precision.\n");
#else
  printf("Using single precision.\n");
#endif
  
  int N;
  string fnS_r;
  string fnS_i;
  string fnInterval;
  int k;
  double delta_o;
  double xtol;
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

  Factorize(H, interval, nthreads);

  return 0;
}

void Factorize(Eigen::SparseMatrix<Scalar, Eigen::ColMajor>  &T,
               std::vector<RealType>                 &intervals,
               int                                     nThreads)
{
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using Eigen::Matrix;
  using Eigen::Vector;
  
  ThreadSafeQueue<std::pair<RealType, RealType>> intervalQ;
  ThreadSafeQueue<std::shared_ptr<SparseLU<SparseMatrix<Scalar>>>> resQ;
  ThreadSafeQueue<RealType> resSQ;
  ThreadSafeQueue<int> sem(nThreads + 3); 

  for (int i = 1; i < intervals.size(); ++i) {
    // (shift, radius)
    intervalQ.push({(intervals[i] + intervals[i - 1]) / 2, (intervals[i] - intervals[i - 1]) / 2});
  }

  printCurrentTime();
  printf(": Start solving. Lauching %d LU worker threads.\n", nThreads);

  auto workerLU = [&]{
    std::pair<RealType, RealType> intvl;
    while (intervalQ.pop(intvl, false)) {
      sem.push(1);
      RealType sigma = intvl.first;

      // Shift the matrix
      SparseMatrix<Scalar> Tshift = T;
      Tshift.diagonal().array() -= sigma;

      // Invert the matrix (finding LU decomposition)
      std::shared_ptr<SparseLU<SparseMatrix<Scalar>>> luP = std::make_shared<SparseLU<SparseMatrix<Scalar>>>();
      luP->isSymmetric(true);
      luP->analyzePattern(Tshift);
      luP->factorize(Tshift);

      resQ.push(luP);
      resSQ.push(sigma);
      
      printCurrentTime();
      printf(": Finished LU at sigma = %f\n", sigma);
    }
  };

  auto workerSave = [&] {
    std::shared_ptr<SparseLU<SparseMatrix<Scalar>>> luP;
    RealType sigma;
    int tmp;

    for (int i = 0; i < intervals.size() - 1; ++i) {
      resQ.pop(luP);
      resSQ.pop(sigma);
      saveLU<Scalar, RealType>(*luP, sigma);
      printCurrentTime();
      printf(": Saved LU at sigma = %f to disk\n", sigma);
      sem.pop(tmp, false);
    }
  };

  // Lauch lu worker threads.
  std::thread svT(workerSave);

  std::vector<std::thread> luTs;
  printf("Lauch luTs\n");
  for (int i = 0; i < nThreads; ++i) {
    luTs.emplace_back(workerLU);
  }

  // Wait for workers to finish.
  for (int i = 0; i < nThreads; ++i) {
    luTs[i].join();
  }
  svT.join();

  printCurrentTime();
  printf(": All LU factorization finished and saved to disk.\n");
}