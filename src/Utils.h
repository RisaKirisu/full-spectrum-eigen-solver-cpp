#pragma once

void printUsage();

template <typename T>
void readArray(std::string file, int size, std::vector<T> &output) {
  std::ifstream fs(file);
  output.clear();
  output.resize(size);

  for (int i = 0; i < size; ++i) {
    fs >> output[i];
  }

  printf("Finished reading %s.\n", file.c_str());
}