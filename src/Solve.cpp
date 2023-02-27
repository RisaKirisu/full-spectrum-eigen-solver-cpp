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
           double                tol);

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
    printf("  %d - Device name: %s\n", i, prop.name);
  }

  Solve(N, interval, k, xtol);

  return 0;
}

void Solve(int                   N,
           std::vector<RealType> &intervals,
           int                   k,
           double                tol)
{
  using Eigen::SparseMatrix;
  using Eigen::SparseLU;
  using Eigen::Matrix;
  using Eigen::Vector;


  // Defaut to use 1/2 of total available memory 
  size_t availMem = getTotalSystemMemory() / 2;
  printf("Avail ram: %lu MB\n", availMem);
  int width = N * N * 2;
  
  std::vector<std::pair<RealType, RealType>> intervalQ;

  for (int i = 1; i < intervals.size(); ++i) {
    // (shift, radius)
    intervalQ.emplace_back((intervals[i] + intervals[i - 1]) / 2, (intervals[i] - intervals[i - 1]) / 2);
  }

  printCurrentTime();
  printf(": Start solving. Lauching GPU worker threads.\n");

  size_t resSize = (width * sizeof(Scalar) + sizeof(RealType)) / 1024; // KB
  int resBufSize = availMem * 1024 / resSize;   // Number of eigenpairs to keep in memory before saving to disk
  // Result buffers
  Vector<RealType, -1> resE(1);
  Matrix<Scalar, -1, -1> resV(width, 1);
  std::vector<int> found;
  std::vector<RealType> sigmas;
  int lastSave = 0;

  for (int i = 0; i < intervalQ.size(); ++i) {
    std::pair<RealType, RealType> itv = intervalQ[i];
    RealType sigma = itv.first;
    RealType radius = itv.second;

    GPU::cusparseLU<Scalar> lu;
    loadLU(lu, sigma);
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
    found.push_back(idx.size());
    sigmas.push_back(sigma);

    
    printCurrentTime();
    if (idx.size() > 0) {
      // Copy results to result buffer.
      printf(": %d Eigenvalues found at sigma = %f. Range:(%e, %e) \n", idx.size(), sigma, E(idx).minCoeff(), E(idx).maxCoeff());
      resE.conservativeResize(resE.rows() + idx.size());
      resE.bottomRows(idx.size()) = E(idx);
      resV.conservativeResize(Eigen::NoChange, resV.cols() + idx.size());
      resV.rightCols(idx.size()) = V(Eigen::placeholders::all, idx);
    } else {
      printf(": 0 Eigenvalues found at sigma = %f.\n", sigma);
    }

    // Save buffered results to file
    if (resE.rows() >= resBufSize || i == intervalQ.size() - 1) {
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
      resV.resize(width, 1);
      found.clear();
      sigmas.clear();
    }
  }

  printCurrentTime();
  printf(": All results saved to disk.\n");
}