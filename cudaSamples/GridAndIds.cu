#include <cuda.h>
#include <stdio.h>

__global__ void printDetails() {
  if (blockIdx.x == 63)
    printf("GridDim : %d, BlockIdx : %d, ThreadIdx.x : %d\n", gridDim.x,
           blockIdx.x, threadIdx.x);
}

int main() {
  cudaSetDevice(1);
  printDetails<<<64, 72>>>();
  cudaDeviceSynchronize();
  return 0;
}