#include "InternalInclude.h"

void printUsage() {
  printf("Usage: ./solve <N> <delta_o> <real part of S> <imaginary part of S> <intervals> <k> <tolarent>(times machine epsilon) <number of threads>\n");
  exit(1);
}

void printCurrentTime() {
  auto now = std::chrono::system_clock::now();
  std::time_t nowt = std::chrono::system_clock::to_time_t(now);
  std::string tstr = std::ctime(&nowt);
  tstr.pop_back();
  printf("[%s]", tstr.c_str());
}

/* Return Total System Memory in MB*/
size_t getTotalSystemMemory() {
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size / 1048576;
}