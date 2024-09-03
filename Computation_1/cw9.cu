#include <driver_types.h>
#include <stdio.h>
#include <time.h>
__global__ void warpCondition(unsigned *vector, unsigned vectorsize) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id % 2)
    vector[id] = id;
  else
    vector[id] = vectorsize * vectorsize;
  vector[id]++;
}
int main() {
  unsigned resultgpu[4096];
  unsigned *dresultgpu;
  for (int i = 0; i < 4096; i++) {
    resultgpu[i] = 0;
  }
  unsigned startgpu = clock();

  cudaMalloc(&dresultgpu, 4096 * sizeof(int));
  cudaMemcpy(dresultgpu, resultgpu, 4096 * sizeof(int), cudaMemcpyHostToDevice);

  warpCondition<<<64, 64>>>(dresultgpu, 4096);
  cudaMemcpy(resultgpu, dresultgpu, 4096 * sizeof(int), cudaMemcpyDeviceToHost);
  unsigned endgpu = clock();

  double gpu_time_used =
      (((double)(endgpu - startgpu)) / CLOCKS_PER_SEC) * 1000;
  printf("time taken by gpu is %f ms\n", gpu_time_used);
  for (int i = 0; i < 4096; i++)
    printf("%d, ", resultgpu[i]);
  printf("\n");
  return 0;
}