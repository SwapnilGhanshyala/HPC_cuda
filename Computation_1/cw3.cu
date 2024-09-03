#include <stdio.h>

__global__ void initArray(int *arr, int len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len)
    arr[id] = 0;
}

__global__ void addId(int *arr, int len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len)
    arr[id] += id;
}

int main() {
  int N = 8000;
  int arr[N], *da;
  cudaMalloc(&da, N * sizeof(int));

  //  Below will not work for N=8000 because blockSize is limited to 1024 on
  //  this pc
  //   initArray<<<1, N>>>(da, N);
  //   cudaDeviceSynchronize();
  //   addId<<<1, N>>>(da, N);
  //   cudaDeviceSynchronize();

  initArray<<<10, N / 10>>>(da, N);
  cudaDeviceSynchronize(); // optional between 2 kernel launches.
  addId<<<10, N / 10>>>(da, N);
  cudaDeviceSynchronize();

  cudaMemcpy(arr, da, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    printf("Id %d : %d\n ", i, arr[i]);
}