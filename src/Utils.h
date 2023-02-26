#pragma once

#define ERROR(message, ...)                 \
{                                           \
  fprintf(stderr, "Error: ");               \
  fprintf(stderr, (message), __VA_ARGS__);  \
  exit(1);                                  \
}

void printSolveUsage();
void printFactorizeUsage();
void printCurrentTime();

/* Return Total System Memory in MB*/
size_t getTotalSystemMemory();

/* Load <size> space sperated values in a plain text file to a vector*/
template <typename T>
void loadFromFile(std::string file, std::vector<T> &output, int size) {
  std::ifstream fs(file);
  if (!fs.is_open()) {
    ERROR("Cannot open %s. Are you sure it exists?\n", file.c_str());
  }

  output.clear();
  output.resize(size);

  for (int i = 0; i < size; ++i) {
    fs >> output[i];
    if (fs.eof() && i < size - 1) {
      ERROR("%s contains less than %d values.\n", file.c_str(), size);
    }
  }

  printf("Finished reading %d elements from %s.\n", size, file.c_str());
}

/* Load all space sperated values in a plain text file to a vector*/
template <typename T>
void loadFromFile(std::string file, std::vector<T> &output) {
  std::ifstream fs(file);
  if (!fs.is_open()) {
    ERROR("Cannot open %s. Are you sure it exists?\n", file.c_str());
  }

  output.clear();
  T cur;
  while (!fs.eof()) {
    fs >> cur;
    output.push_back(cur);
  }
  output.shrink_to_fit();

  printf("Finished reading %d elements from %s.\n", output.size(), file.c_str());
}

/* A simple thread safe queue*/
template<typename T>
class ThreadSafeQueue {
public:
  ThreadSafeQueue() : m_max_capacity(-1) {}
  explicit ThreadSafeQueue(int max_capacity) : m_max_capacity(max_capacity) {}

  void push(const T& value) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_max_capacity != -1 && m_queue.size() >= m_max_capacity) {
      m_cv_cap.wait(lock, [&]() { return m_max_capacity == -1 || m_queue.size() < m_max_capacity; });
    }
    m_queue.push(value);
    lock.unlock();
    m_cv_emp.notify_one();
  }

  bool pop(T& value, bool blocking = true) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (blocking && m_queue.empty()) {
      m_cv_emp.wait(lock, [&]() { return !m_queue.empty(); });
    }
    if (m_queue.empty()) {
      return false;
    }
    value = std::move(m_queue.front());
    m_queue.pop();
    lock.unlock();
    m_cv_cap.notify_one();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.empty();
  }

private:
  std::queue<T> m_queue;
  mutable std::mutex m_mutex;
  mutable std::condition_variable m_cv_cap;
  mutable std::condition_variable m_cv_emp;
  const int m_max_capacity;
};

// Requires c++17
/* Save an Eigen::SparseLU to files. */
template <typename Scalar, typename RealType>
void saveLU(Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> &lu, RealType sigma) {
  namespace fs = std::filesystem;
  std::string dirname = "lu-" + std::to_string(sigma);
  if (!fs::create_directory(dirname)) {
    fprintf(stderr, "Cannot create directory %s.\n", dirname.c_str());
    exit(1);
  }
  std::string fnLdata = dirname + "/Ldata.npy";
  std::string fnLrowptr = dirname + "/Lrptr.npy";
  std::string fnLcolidx = dirname + "/Lcidx.npy";
  std::string fnUdata = dirname + "/Udata.npy";
  std::string fnUrowptr = dirname + "/Urptr.npy";
  std::string fnUcolidx = dirname + "/Ucidx.npy";
  std::string fnPermR = dirname + "/perm_r.npy";
  std::string fnPermCI = dirname + "/perm_cI.npy";

  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> L, U;
  lu.getCsrLU(L, U);

  unsigned long shape[1] = {L.nonZeros()};
  npy::SaveArrayAsNumpy(fnLdata, false, 1, shape, L.valuePtr());
  npy::SaveArrayAsNumpy(fnLcolidx, false, 1, shape, L.innerIndexPtr());
  shape[0] = L.rows() + 1;
  npy::SaveArrayAsNumpy(fnLrowptr, false, 1, shape, L.outerIndexPtr());

  shape[0] = {U.nonZeros()};
  npy::SaveArrayAsNumpy(fnUdata, false, 1, shape, U.valuePtr());
  npy::SaveArrayAsNumpy(fnUcolidx, false, 1, shape, U.innerIndexPtr());
  shape[0] = U.rows() + 1;
  npy::SaveArrayAsNumpy(fnUrowptr, false, 1, shape, U.outerIndexPtr());

  Eigen::Vector<int, -1> perm = lu.rowsPermutation().indices();
  shape[0] = {perm.rows()};
  npy::SaveArrayAsNumpy(fnPermR, false, 1, shape, perm.data());

  perm = lu.colsPermutation().inverse().eval().indices();
  shape[0] = {perm.rows()};
  npy::SaveArrayAsNumpy(fnPermCI, false, 1, shape, perm.data());
}