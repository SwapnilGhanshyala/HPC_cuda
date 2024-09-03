#include <cmath>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#define N 4000
struct Point {
  int x, y;
} arr[N];

__global__ void calcAVG(unsigned *darrx, unsigned *darry, float *globalAVG,
                        unsigned *globalSum, unsigned size) {
  unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int first = threadId * 4, last = threadId * 4 + 3;
  unsigned sum = 0;
  for (int i = first; i < size && i <= last; i++) {
    sum += darrx[i];
  }
  atomicAdd(globalAVG, (float)sum / size);
  __syncthreads(); // it needs synchronization across TBlocks and not
                   // __syncthreads
  bool flag = false;
  for (int i = first; i < size && i <= last; i++)
    if (darry[i] > *globalAVG) {
      flag = true;
      break;
    }
  if (flag)
    for (int i = first; i < size && i <= last; i++)
      darry[i] = *globalAVG;
  else
    for (int i = first; i < size && i <= last; i++)
      (*globalSum) += darry[i];
}
int main() {
  int arrx[N];
  int arry[N];
  for (int i = 0; i < N; i++) {
    arrx[i] = arr[i].x = rand() % 10;
    arry[i] = arr[i].y = rand() % 10;
  }

  unsigned *darrx, *darry, *globalSum;
  float *globalAVG;
  cudaMalloc(&darrx, sizeof(int) * N);
  cudaMalloc(&darry, sizeof(int) * N);
  cudaMalloc(&globalAVG, sizeof(float));
  cudaMalloc(&globalSum, sizeof(int));
  cudaMemcpy(darrx, arrx, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(darry, arry, N * sizeof(int), cudaMemcpyHostToDevice);
  unsigned gridSize =
      static_cast<unsigned>(ceil(static_cast<double>(N) / 128.0));
  calcAVG<<<gridSize, 32>>>(darrx, darry, globalAVG, globalSum, N);
  cudaDeviceSynchronize();
  unsigned *rarry, *rglobalSum;
  float *rglobalAVG;
  cudaMemcpy(rarry, darry, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(rglobalAVG, globalAVG, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(rglobalSum, globalSum, sizeof(int), cudaMemcpyDeviceToHost);
  printf("globalAVG is %f\n", *rglobalAVG);
  printf("globalSum is %u \n", *rglobalSum);

  return 0;
}