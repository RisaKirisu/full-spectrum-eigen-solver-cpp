#include <iostream>
#include <complex>
#include <limits>
using namespace std;

int main() {
  double e = numeric_limits<float>::epsilon();
  cout << "eps: " << e << endl;
  return 0;
}
