#include <cuda_runtime.h>
#include <iostream>

int main() {
  int device;
  cudaGetDevice(&device);
  std::cout << "Default CUDA device is device " << device << std::endl;

  return 0;
}
