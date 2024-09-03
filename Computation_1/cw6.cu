#include <stdio.h>

#define N 5
#define M 6
__global__ void init2DArray(unsigned *m) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  m[id] = id;
}
int main() {
  unsigned arr[N * M], *dmatrix;
  cudaMalloc(&dmatrix, N * M * sizeof(unsigned));
  init2DArray<<<N, M>>>(dmatrix);
  cudaMemcpy(arr, dmatrix, N * M * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      printf("%d , ", arr[i * M + j]);
  printf("\n");
  return 0;
}