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
           int                                     nThreads,
           int                                     nDevice);

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
    printSolveUsage();
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

  // Collect system info
  int nDevice = 0;
  CHECK_CUDA( cudaGetDeviceCount(&nDevice) );
  nthreads = (nthreads == -1) ? std::thread::hardware_concurrency() : nthreads;
  nthreads -= nDevice + 1;    // 1 thread for each GPU + 1 thread for handling results
  nthreads = std::max((int) (nthreads / 2.5), 1);

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
  printCurrentTime();
  printf(": Finished constructing hamiltonian of size %d * %d.\n", N, N);


  Solve(H, interval, k, xtol, nthreads, nDevice);

  return 0;
}

void Solve(Eigen::SparseMatrix<Scalar, Eigen::ColMajor>  &T,
           std::vector<RealType>                 &intervals,
           int                                            k,
           double                                       tol,
           int                                     nThreads,
           int                                     nDevice)
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
  ThreadSafeQueue<std::tuple<std::shared_ptr<Matrix<Scalar, -1, -1>>, std::shared_ptr<Vector<RealType, -1>>, RealType, RealType>> resQ;

  for (int i = 1; i < intervals.size(); ++i) {
    // (shift, radius)
    intervalQ.push({(intervals[i] + intervals[i - 1]) / 2, (intervals[i] - intervals[i - 1]) / 2});
  }

  printCurrentTime();
  printf(": Start solving. Lauching %d LU worker threads and %d GPU worker threads.\n", nThreads, nDevice);

  auto workerLU = [&]{
    std::pair<RealType, RealType> intvl;
    while (intervalQ.pop(intvl, false)) {
      stagQ.push(true);
      RealType sigma = intvl.first;

      // Shift the matrix
      SparseMatrix<Scalar> Tshift = T;
      Tshift.diagonal().array() -= sigma;

      // Invert the matrix (finding LU decomposition)
      std::shared_ptr<SparseLU<SparseMatrix<Scalar>>> luP = std::make_shared<SparseLU<SparseMatrix<Scalar>>>();
      luP->isSymmetric(true);
      luP->analyzePattern(Tshift);
      luP->factorize(Tshift);

      invQ.push(std::make_tuple(luP, sigma, intvl.second));
      printCurrentTime();
      printf(": Finished LU at sigma = %f\n", sigma);
    }
  };

  auto workerEig = [&] (int device) {
    CHECK_CUDA( cudaSetDevice(device) );

    bool s;
    std::tuple<std::shared_ptr<SparseLU<SparseMatrix<Scalar>>>, RealType, RealType> work;
    
    for (int i = 0; i < intervals.size() - 1; ++i) {
      invQ.pop(work);
      stagQ.pop(s, false);

      RealType sigma = std::get<1>(work);
      RealType radius = std::get<2>(work);
      // Calculate eigenvalues
      GPU::cusparseLU<Scalar> lu(*std::get<0>(work));
      GPU::Eigsh<Scalar> eigsh(lu);
      eigsh.solve(k, GPU::LM, 0, 0, tol * std::numeric_limits<RealType>::epsilon());
      
      std::shared_ptr<Vector<RealType, -1>> pE = std::make_shared<Vector<RealType, -1>>();
      std::shared_ptr<Matrix<Scalar, -1, -1>> pV = std::make_shared<Matrix<Scalar, -1, -1>>();
      *pE = eigsh.eigenvalues();
      *pV = eigsh.eigenvectors();
      resQ.push(std::make_tuple(pV, pE, sigma, radius));

      printCurrentTime();
      printf(": Finished solving at sigma = %f.\n", sigma);
    }
  };

  auto workerRes = [&] {
    std::tuple<std::shared_ptr<Matrix<Scalar, -1, -1>>, std::shared_ptr<Vector<RealType, -1>>, RealType, RealType> res;

    size_t resSize = (T.rows() * sizeof(Scalar) + sizeof(RealType)) / 1024; // KB
    int resBufSize = availMem * 1024 / 2 / resSize;   // Number of eigenpairs to keep in memory before saving to disk

    // Result buffers
    Vector<RealType, -1> resE(1);
    Matrix<Scalar, -1, -1> resV(T.rows(), 1);
    std::vector<int> found;
    std::vector<RealType> sigmas;
    int lastSave = 0;

    for (int i = 0; i < intervals.size() - 1; ++i) {
      resQ.pop(res);
      std::shared_ptr<Matrix<Scalar, -1, -1>> pV = std::get<0>(res);
      std::shared_ptr<Vector<RealType, -1>> pE = std::get<1>(res);
      RealType sigma = std::get<2>(res);
      RealType radius = std::get<3>(res);

      sigmas.push_back(sigma);

      // Invert and shift back eigenvalues
      *pE = pE->cwiseInverse();
      pE->array() += sigma;

      // Filter out results outside of desired interval
      std::vector<int> idx;
      for (int j = 0; j < pE->rows(); ++j) {
        if ((*pE)(j) > sigma - radius && (*pE)(j) < sigma + radius) {
          idx.push_back(j);
        }
      }

      // Retry with more ncv.
      // if (idx.size() == 0) {
      //   eigsh.solve(k, GPU::LM, k * 3, 0, tol * std::numeric_limits<RealType>::epsilon());
      //   E = eigsh.eigenvalues();
      //   V = eigsh.eigenvectors();
      //   for (int j = 0; j < E.rows(); ++j) {
      //     if (E(j) > sigma - radius && E(j) < sigma + radius) {
      //       idx.push_back(j);
      //     }
      //   }
      // }
      found.push_back(idx.size());
      printf("%d Eigenvalues found at sigma = %f. Range:(%e, %e) \n", idx.size(), sigma, (*pE)(idx).minCoeff(), (*pE)(idx).maxCoeff());

      if (idx.size() > 0) {
        // Copy results to result buffer.
        resE.conservativeResize(resE.rows() + idx.size());
        resE.bottomRows(idx.size()) = (*pE)(idx);
        resV.conservativeResize(Eigen::NoChange, resV.cols() + idx.size());
        resV.rightCols(idx.size()) = (*pV)(Eigen::placeholders::all, idx);
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
        fnFound += std::to_string(i);
        fnFound += ".npy";
        std::string fnSigma = "Sigma_";
        fnSigma += std::to_string(i);
        fnSigma += ".npy";
        
        unsigned long shapeE[1] = {resE.rows() - 1};
        npy::SaveArrayAsNumpy(fnE, false, 1, shapeE, resE.data() + 1);

        unsigned long shapeV[2] = {resV.rows(), resV.cols() - 1};
        npy::SaveArrayAsNumpy(fnV, true, 2, shapeV, resV.data() + resV.rows());

        unsigned long shapeF[1] = {found.size()};
        npy::SaveArrayAsNumpy(fnFound, false, 1, shapeF, found.data());

        unsigned long shapeS[1] = {sigmas.size()};
        npy::SaveArrayAsNumpy(fnSigma, false, 1, shapeS, sigmas.data());

        printCurrentTime();
        printf(": Saved %d intervals to file.\n", i - lastSave + 1);
        lastSave = i;

        resE.resize(1);
        resV.resize(T.rows(), 1);
        found.clear();
        sigmas.clear();
      }
    }
  };

  // Lauch lu worker threads.
  std::vector<std::thread> luTs;
  printf("Lauch luTs\n");
  for (int i = 0; i < nThreads; ++i) {
    luTs.emplace_back(workerLU);
  }

  // Lauch eigen worker thread
  std::vector<std::thread> eigTs;
  printf("Lauch eigTs\n");
  for (int i = 0; i < nDevice; ++i) {
    eigTs.emplace_back(workerEig, i);
  }

  std::thread resT(workerRes);

  // Wait for workers to finish.
  for (int i = 0; i < nThreads; ++i) {
    luTs[i].join();
  }
  printCurrentTime();
  printf(": All LU factorization finished.\n");

  for (int i = 0; i < nDevice; ++i) {
    eigTs[i].join();
  }
  printCurrentTime();
  printf(": All solver finished.\n");

  resT.join();
  printCurrentTime();
  printf(": All results saved to disk.\n");

  return;
}