#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void K(int *x) {
  *x = 0;
  printf("x = %d\n", *x);
}
int main() {
  int *x;
  K<<<2, 10>>>(x);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  printf("error=%d, %s, %s\n", err, cudaGetErrorName(err),
         cudaGetErrorString(err));
  return 0;
}