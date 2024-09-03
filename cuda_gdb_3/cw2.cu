#include <stdio.h>

__global__ void K(int *p) {
  *p = 0;
  printf("%d\n", *p);
}

int main() {
  int *x, *y;
  cudaMalloc(&x, sizeof(int));
  K<<<2, 10>>>(x);
  y = x;
  cudaFree(y);
  K<<<2, 10>>>(x);
  cudaDeviceSynchronize();
  return 0;
}