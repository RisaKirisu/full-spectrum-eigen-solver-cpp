#include "InternalIncludeCuda.h"
// #include "InternalInclude.h"

#ifdef USE_DOUBLE
typedef std::complex<double> Scalar;
typedef double RealType;
#else
typedef std::complex<float> Scalar;
typedef float RealType;
#endif

void Solve(Eigen::SparseMatrix<Scalar, Eigen::ColMajor>  &T,
           std::vector<RealType>                 &intervals,
           int                                            k,
           double                                       tol,
           int                                     nThreads);

int main(int argc, char *argv[]) {
  using Eigen::Matrix;
  using Eigen::Vector;
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using Eigen::Dynamic;
  using std::string;
  using std::cout;
  using std::cin;
  using std::endl;
  using std::complex;
  
  int N;
  string fnS_r;
  string fnS_i;
  string fnInterval;
  int k;
  double delta_o;
  double xtol;
  int nthreads;

  // Get commandline arguments.
  if (argc != 9) {
    printUsage();
  }
  N = std::stoi(argv[1]);
  delta_o = std::stod(argv[2]);
  fnS_r = string(argv[3]);
  fnS_i = string(argv[4]);
  fnInterval = string(argv[5]);
  k = std::stoi(argv[6]);
  xtol = std::stod(argv[7]);
  nthreads = std::stoi(argv[8]);
  printf("Args: %d, %f, %s, %s, %s, %d, %f, %d\n", N, delta_o, fnS_r.c_str(), fnS_i.c_str(), fnInterval.c_str(), k, xtol, nthreads);

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
  // std::vector<Scalar> S;
  // for (int i = 0; i < N * N; ++i) {
  //   S.emplace_back(1, 0);
  // }
  GetHamiltonian(N, S, 0.06, H);

  Solve(H, interval, k, xtol, nthreads);

  return 0;
}

void Solve(Eigen::SparseMatrix<Scalar, Eigen::ColMajor>  &T,
           std::vector<RealType>                 &intervals,
           int                                            k,
           double                                       tol,
           int                                     nThreads)
{
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using Eigen::Matrix;
  using Eigen::Vector;


  // Defaut to use 1/2 of total available memory 
  size_t availMem = getTotalSystemMemory() / 2;
  printf("Avail ram: %lu MB\n", availMem);
  size_t luSize;
  {
    SparseLU<SparseMatrix<Scalar>> lu;
    lu.isSymmetric(true);
    lu.analyzePattern(T);
    lu.factorize(T);
    luSize = ((lu.nnzL() + lu.nnzU()) * (sizeof(Scalar) + sizeof(int)) + lu.rows() * sizeof(int)) / 1048576;
  }
  printf("LU size: %lu MB\n", luSize);

  int maxAmountLU = availMem / luSize;
  printf("max: %d\n", maxAmountLU);
  
  ThreadSafeQueue<std::pair<RealType, RealType>> intervalQ;
  ThreadSafeQueue<bool> stagQ(maxAmountLU);   // Emulate a semaphore. Prevent starting of new lu job after max is reached (prevent memory overflow). 
  ThreadSafeQueue<std::tuple<std::shared_ptr<SparseLU<SparseMatrix<Scalar>>>, RealType, RealType>> invQ;

  for (int i = 1; i < intervals.size(); ++i) {
    // (shift, radius)
    intervalQ.push({(intervals[i] + intervals[i - 1]) / 2, (intervals[i] - intervals[i - 1]) / 2});
  }

  printCurrentTime();
  printf(": Start solving. Using %d threads.\n", nThreads);

  auto workerLU = [&]{
    std::pair<RealType, RealType> intvl;
    while (intervalQ.pop(intvl, false)) {
      stagQ.push(true);
      RealType sigma = intvl.first;

      SparseMatrix<Scalar> Tshift = T;
      Tshift.diagonal().array() -= sigma;

      std::shared_ptr<SparseLU<SparseMatrix<Scalar>>> luP = std::make_shared<SparseLU<SparseMatrix<Scalar>>>();
      luP->isSymmetric(true);
      luP->analyzePattern(Tshift);
      luP->factorize(Tshift);

      invQ.push(std::make_tuple(luP, sigma, intvl.second));
      printCurrentTime();
      printf(": Finished LU at sigma = %f\n", sigma);
    }
  };

  auto workerEig = [&] {
    bool s;
    size_t resSize = (T.rows() * sizeof(Scalar) + sizeof(RealType)) / 1024; // KB
    int resBufSize = availMem * 1024 / 2 / resSize;   // Number of eigenpairs to keep in memory before saving to disk

    // Result buffers
    Vector<RealType, -1> resE(1);
    Matrix<Scalar, -1, -1> resV(T.rows(), 1);
    std::vector<int> found;
    std::vector<RealType> sigmas;

    std::tuple<std::shared_ptr<SparseLU<SparseMatrix<Scalar>>>, RealType, RealType> work;
    
    for (int i = 0; i < intervals.size() - 1; ++i) {
      invQ.pop(work);
      stagQ.pop(s, false);

      RealType sigma = std::get<1>(work);
      RealType radius = std::get<2>(work);
      sigmas.push_back(sigma);
      // Calculate eigenvalues
      GPU::cusparseLU<Scalar> lu(*std::get<0>(work));
      GPU::Eigsh<Scalar> eigsh(lu);
      eigsh.solve(k, GPU::LM, 0, 0, tol * std::numeric_limits<RealType>::epsilon());
      Vector<RealType, -1> E = eigsh.eigenvalues();
      Matrix<Scalar, -1, -1> V = eigsh.eigenvectors();

      // Invert and shift back eigenvalues
      E = E.cwiseInverse();
      E.array() += sigma;

      // Filter out results outside of desired interval
      std::vector<int> idx;
      for (int j = 0; j < E.rows(); ++j) {
        if (E(j) > sigma - radius && E(j) < sigma + radius) {
          idx.push_back(j);
        }
      }

      // Retry with more ncv.
      if (idx.size() == 0) {
        eigsh.solve(k, GPU::LM, k * 3, 0, tol * std::numeric_limits<RealType>::epsilon());
        E = eigsh.eigenvalues();
        V = eigsh.eigenvectors();
        for (int j = 0; j < E.rows(); ++j) {
          if (E(j) > sigma - radius && E(j) < sigma + radius) {
            idx.push_back(j);
          }
        }
      }
      found.push_back(idx.size());

      if (idx.size() > 0) {
        // Copy results to result buffer.
        resE.conservativeResize(resE.rows() + idx.size());
        resE.bottomRows(idx.size()) = E(idx);
        resV.conservativeResize(Eigen::NoChange, resV.cols() + idx.size());
        resV.rightCols(idx.size()) = V(Eigen::placeholders::all, idx);
      }

      // Save buffered results to file
      if (resE.rows() >= resBufSize || i == intervals.size() - 2) {
        std::string fnE = "E_";
        fnE += std::to_string(i);
        fnE += ".npy";
        std::string fnV = "V_";
        fnV += std::to_string(i);
        fnV += ".npy";
        std::string fnFound = "Found_";
        fnV += std::to_string(i);
        fnV += ".npy";
        std::string fnSigma = "Sigma_";
        fnV += std::to_string(i);
        fnV += ".npy";
        
        unsigned long shapeE[1] = {resE.rows() - 1};
        npy::SaveArrayAsNumpy(fnE, false, 1, shapeE, resE.data() + 1);

        unsigned long shapeV[2] = {resV.rows(), resV.cols() - 1};
        npy::SaveArrayAsNumpy(fnV, true, 2, shapeV, resV.data() + resV.rows());

        unsigned long shapeF[1] = {found.size()};
        npy::SaveArrayAsNumpy(fnFound, false, 1, shapeF, found.data());

        unsigned long shapeS[1] = {sigmas.size()};
        npy::SaveArrayAsNumpy(fnSigma, false, 1, shapeS, sigmas.data());

        resE.resize(1);
        resV.resize(T.rows(), 1);
        found.clear();
        sigmas.clear();
      }

      printCurrentTime();
      printf(": Finished solving at sigma = %f. Obtained %d eigval.\n", sigma, idx.size());
    }
  };

  // Lauch lu worker threads.
  std::vector<std::thread> luTs;
  printf("Lauch luTs\n");
  for (int i = 0; i < nThreads - 1; ++i) {
    luTs.emplace_back(workerLU);
  }

  // Lauch eigen worker thread
  printf("Lauch eigT\n");
  std::thread eigT(workerEig);

  // Wait for workers to finish.
  eigT.join();

  for (int i = 0; i < nThreads - 1; ++i) {
    luTs[i].join();
  }

}