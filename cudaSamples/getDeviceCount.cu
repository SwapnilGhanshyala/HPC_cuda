#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
              << " -> " << cudaGetErrorString(error_id) << std::endl;
    std::cout << "Result = FAIL" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "There are " << deviceCount
            << " CUDA capable devices on your system." << std::endl;
  return EXIT_SUCCESS;
}
