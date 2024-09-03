#include <stdio.h>

#define N 5
#define M 6

__global__ void init2DArray(unsigned *arr) {
  int id = threadIdx.x * blockDim.y + threadIdx.y;
  arr[id] = id;
}

int main() {
  dim3 block(N, M, 1);
  unsigned arr[N * M], *dmatrix;
  cudaMalloc(&dmatrix, N * M * sizeof(unsigned));
  init2DArray<<<1, block>>>(dmatrix);
  cudaMemcpy(arr, dmatrix, N * M * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < N * M; i++)
    printf("%d , ", arr[i]);
  printf("\n");
  return 0;
}