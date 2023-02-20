#pragma once

#define ERROR(message, ...)                 \
{                                           \
  fprintf(stderr, "Error: ");               \
  fprintf(stderr, (message), __VA_ARGS__);  \
  exit(1);                                  \
}

void printUsage();
void printCurrentTime();

/* Return Total System Memory in MB*/
size_t getTotalSystemMemory();

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

  bool isFull(bool blocking=true) const {
    std::unique_lock<std::mutex> lock(m_mutex);
    // printf("isFull\n");
    if (m_max_capacity == -1 || m_queue.size() < m_max_capacity) {
      return false;
    }
    // printf("??? %d, %d\n", m_max_capacity, m_queue.size());
    if (blocking) {
      m_cv_cap.wait(lock, [&]() { return m_max_capacity == -1 || m_queue.size() < m_max_capacity; });
    }
    return m_queue.size() < m_max_capacity;
  }

  int size() const { return m_queue.size(); }

private:
  std::queue<T> m_queue;
  mutable std::mutex m_mutex;
  mutable std::condition_variable m_cv_cap;
  mutable std::condition_variable m_cv_emp;
  const int m_max_capacity;
};