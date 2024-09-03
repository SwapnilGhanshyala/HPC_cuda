#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#define N 100

__global__ void cw2(int *d_a) {
  if (threadIdx.x < N)
    d_a[threadIdx.x] = threadIdx.x * threadIdx.x;
}
int main() {
  int a[N], *d_a;
  cudaMalloc(&d_a, N * sizeof(int));
  cw2<<<1, N>>>(d_a);
  cudaDeviceSynchronize(); // optional
  cudaMemcpy(a, d_a, N * sizeof(int),
             cudaMemcpyDeviceToHost); // O(N) operation theoretically
  // Opt 1: it can get data in blocks instead of element by element, since block
  // size is constant then still O(N)
  // Opt 2: latency hiding using cudaMemcpyAsync
  printf("Printing on host\n");
  for (int i = 0; i < N; i++)
    printf("%d\n", a[i]);
  return 0;
}