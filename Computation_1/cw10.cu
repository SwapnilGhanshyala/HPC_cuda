#include <stdio.h>
#include <time.h>
// code for dod = 4
__global__ void dod4(unsigned *vector, unsigned vectorsize) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < vectorsize)
    switch (id % 4) {
    case 0:
      vector[id] = 0;
      break;
    case 1:
      vector[id] = 1;
      break;
    case 2:
      vector[id] = 2;
      break;
    case 3:
      vector[id] = 3;
      break;
    }
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

  dod4<<<64, 64>>>(dresultgpu, 4096);
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