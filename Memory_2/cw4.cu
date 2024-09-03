#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#define BLOCKSIZE 1024
// if threa are no syncthreads, there is no need for a warp of the threadblock
// to wait up on other threads of the TBlock to be at the same instruction. So
// first warp that has threads of id 0 to 31 will execute condition 1 and
// condition 2 and then the print statement.
// this will happen due to lack of synchronization between the threads of the
// thread Block. so there is a chance that there will be a "s=1" as output. due
// to __syncthreads, all threads of the block will synchronize whenever the it
// is encountered. so it is guaranteed that warp 0 will not start executing
// condition 2 untill all other warps reach the same point.
// Although the syncthread between condition 1 and cond2 are not needed.
__global__ void dkernel() {
  __shared__ unsigned s;
  if (threadIdx.x == 0)
    s = 0;
  // __syncthreads();
  if (threadIdx.x == 1)
    s += 1;
  // __synchreads();
  if (threadIdx.x == 100)
    s += 2;
  // __syncthreads();
  if (threadIdx.x == 0)
    printf("s=%d,", s);
}
int main() {
  for (int i = 0; i < 36000; i++) {
    dkernel<<<2, BLOCKSIZE>>>();
    cudaDeviceSynchronize();
  }
  return 0;
}