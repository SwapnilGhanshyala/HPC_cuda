#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#define BLOCKSIZE 34
// if BLOCKSIZE is < warp size, then there is a automatic synchronization
// between warps(single in this case) in the threadBlock, therefore no need for
// __syncthreads. syncthreads is needed is size is greater.
__global__ void dkernel() {
  __shared__ char str[BLOCKSIZE + 1];
  str[threadIdx.x] = 'A' + (threadIdx.x + blockIdx.x) % BLOCKSIZE;
  if (threadIdx.x == 0) {
    str[BLOCKSIZE] = '\0';
  }
  __syncthreads();
  // without syncthreads, there is no guarantee that other warps of the thread
  // block have finished writing to the shared location
  if (threadIdx.x == 0) {
    printf("%d:%s\n", blockIdx.x, str);
  }
}
int main() {
  dkernel<<<10, BLOCKSIZE>>>();
  cudaDeviceSynchronize();
}