#include "InternalInclude.h"

void printSolveUsage() {
  printf("Usage: ./solve <N> <intervals> <k> <tolarent>(times machine epsilon)\n");
  exit(1);
}

void printFactorizeUsage() {
  printf("Usage: ./factorize <N> <delta_o> <real part of S> <imaginary part of S> <intervals> <number of threads>\n");
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

void setThreadScheduling(std::thread &th, int policy, int priority) {
  sched_param sch_params;
  sch_params.sched_priority = priority;
  if(pthread_setschedparam(th.native_handle(), policy, &sch_params)) {
    std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
  }
}
