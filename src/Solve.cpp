#include "InternalIncludeCuda.h"

// #include "InternalInclude.h"

#ifdef USE_DOUBLE
typedef std::complex<double> Scalar;
typedef double RealType;
#else
typedef std::complex<float> Scalar;
typedef float RealType;
#endif

void Solve(int                   N,
           std::vector<RealType> &intervals,
           int                   k,
           double                tol,
           int                   nDevice);

int main(int argc, char *argv[]) {
  using std::string;
  
  int N;
  string fnInterval;
  int k;
  double xtol;

  // Get commandline arguments.
  if (argc != 5) {
    printSolveUsage();
  }
  N = std::stoi(argv[1]);
  fnInterval = string(argv[2]);
  k = std::stoi(argv[3]);
  xtol = std::stod(argv[4]);
  printf("Args: %d, %s, %d, %f\n", N, fnInterval.c_str(), k, xtol);

  std::vector<RealType> interval;
  loadFromFile(fnInterval, interval);

  // Get GPU info
  int nDevice;
  CHECK_CUDA( cudaGetDeviceCount(&nDevice) );
  printf("Available GPU count: %d\n", nDevice);
  for (int i = 0; i < nDevice; ++i) {
    cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties(&prop, i) );
    printf("  %d - Device name: %s  ", i, prop.name);
    printf("  Avail VRAM: %lu\n", prop.totalGlobalMem / 1024 / 1024);
  }

  Solve(N, interval, k, xtol, nDevice);

  return 0;
}

void Solve(int                   N,
           std::vector<RealType> &intervals,
           int                   k,
           double                tol,
           int                   nDevice)
{
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using Eigen::Matrix;
  using Eigen::Vector;

  int width = N * N * 2;
  
  ThreadSafeQueue<std::pair<RealType, RealType>> intervalQ;
  ThreadSafeQueue<std::tuple<std::shared_ptr<Matrix<Scalar, -1, -1>>, std::shared_ptr<Vector<RealType, -1>>, RealType, RealType>> resQ;


  for (int i = 1; i < intervals.size(); ++i) {
    // (shift, radius)
    intervalQ.push(std::make_pair((intervals[i] + intervals[i - 1]) / 2, (intervals[i] - intervals[i - 1]) / 2));
  }

  printCurrentTime();
  printf(": Start solving. Lauching GPU worker threads.\n");

  auto workerEig = [&] (int device) {
    CHECK_CUDA( cudaSetDevice(device) );
    cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties(&prop, device) );
    printf("%d - Device name: %s starts working.\n", device, prop.name);

    std::pair<RealType, RealType> itv;
    
    while (intervalQ.pop(itv, false)) {
      RealType sigma = itv.first;
      RealType radius = itv.second;
      // Calculate eigenvalues
      GPU::cusparseLU<Scalar> lu;
      loadLU(lu, sigma);
      GPU::Eigsh<Scalar> eigsh(lu);
      eigsh.solve(k, GPU::LM, 0, 0, tol * std::numeric_limits<RealType>::epsilon());

      printCurrentTime();
      printf(": Finished solving at sigma = %f.\n", sigma);
      
      std::shared_ptr<Vector<RealType, -1>> pE = std::make_shared<Vector<RealType, -1>>();
      std::shared_ptr<Matrix<Scalar, -1, -1>> pV = std::make_shared<Matrix<Scalar, -1, -1>>();
      *pE = eigsh.eigenvalues();
      *pV = eigsh.eigenvectors();
      resQ.push(std::make_tuple(pV, pE, sigma, radius));
    }
  };

  auto workerSave = [&] {
    std::tuple<std::shared_ptr<Matrix<Scalar, -1, -1>>, std::shared_ptr<Vector<RealType, -1>>, RealType, RealType> res;

    int resBufSize = 2000;   // Number of eigenpairs to keep in memory before saving to disk

    // Result buffers
    Vector<RealType, -1> resE(1);
    Matrix<Scalar, -1, -1> resV(width, 1);
    std::vector<int> found;
    std::vector<RealType> sigmas;
    int lastSave = -1;

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

      found.push_back(idx.size());

      if (idx.size() > 0) {
        // Copy results to result buffer.
        printf("%d  Eigenvalues found at sigma  = %f. Range:(%e, %e) \n", idx.size(), sigma, (*pE)(idx).minCoeff(), (*pE)(idx).maxCoeff());
        resE.conservativeResize(resE.rows() + idx.size());
        resE.bottomRows(idx.size()) = (*pE)(idx);
        resV.conservativeResize(Eigen::NoChange, resV.cols() + idx.size());
        resV.rightCols(idx.size()) = (*pV)(Eigen::placeholders::all, idx);
      } else {
        printf("0  Eigenvalues found at sigma  = %f.\n", sigma);
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
        printf(": Saved %d intervals to file.\n", i - lastSave);
        lastSave = i;

        resE.resize(1);
        resV.resize(width, 1);
        found.clear();
        sigmas.clear();
      }
    }
  };

  // Launch an eig thread for each GPU
  std::vector<std::thread> eigTs;
  for (int i = 0; i < nDevice; ++i) {
    eigTs.emplace_back(workerEig, i);
  }

  std::thread saveT(workerSave);

  // Wait for threads to finish.
  for (int i = 0; i < nDevice; ++i) {
    eigTs[i].join();
  }
  saveT.join();

  printCurrentTime();
  printf(": All results saved to disk.\n");
}