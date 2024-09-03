#include <cuda_runtime.h>
#include <iostream>

int main() {
  int device = 0; // You can change this to the device index you want to query
  cudaDeviceProp deviceProp;

  cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, device);
  if (error_id != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed! Error: "
              << cudaGetErrorString(error_id) << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Warp size: " << deviceProp.warpSize << std::endl;

  return EXIT_SUCCESS;
}
