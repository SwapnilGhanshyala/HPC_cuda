// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <stdio.h>
// __global__ int N = 0;
__global__ void dkernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.y == 0 && threadIdx.z == 0 && blockIdx.z == 0)
    printf("%d, %d, %d, %d, %d, %d \n", gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
}
int main() {
  //   int N = 8000;
  dim3 grid(2, 5, 1);
  dim3 block(8, 10, 10);
  dkernel<<<grid, block>>>();
  cudaDeviceSynchronize();
  return 0;
}