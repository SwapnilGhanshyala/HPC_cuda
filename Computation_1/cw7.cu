// Large amount of data
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
__global__ void init2DArray(unsigned *m, unsigned len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len)
    m[id] = id;
}
#define BLOCKSIZE 1024
int main(int nn, char *str[]) {
  unsigned N = atoi(str[1]);
  unsigned *vector, *hvector;
  cudaMalloc(&vector, N * sizeof(unsigned));
  hvector = (unsigned *)malloc(N * sizeof(unsigned));

  unsigned nblocks = ceil(
      (float)N / BLOCKSIZE); // use floating point division not integer division
  printf("nblocks = %d\n", nblocks);
  init2DArray<<<nblocks, BLOCKSIZE>>>(vector, N);
  cudaMemcpy(hvector, vector, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (unsigned i = 0; i < N; i++)
    printf("%d , ", hvector[i]);
  return 0;
}