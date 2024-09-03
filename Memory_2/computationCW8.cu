#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <time.h>

__global__ void squaregpu1(unsigned *matrix, unsigned *result) {
  unsigned ii = blockDim.x * blockIdx.x;
  unsigned j = threadIdx.x;
  unsigned matrixsize = blockDim.x;
  for (unsigned k = 0; k < matrixsize; k++)
    result[ii + j] += matrix[ii + k] * matrix[k * blockDim.x + j];
}
// __global__ void squaregpu1(unsigned *matrix, unsigned *result) {
//   unsigned row = blockIdx.x;
//   unsigned col = threadIdx.x;
//   unsigned matrixsize = blockDim.x;
//   unsigned sum = 0;
//   for (unsigned k = 0; k < matrixsize; k++) {
//     sum += matrix[row * matrixsize + k] * matrix[k * matrixsize + col];
//   }
//   result[row * matrixsize + col] = sum;
// }
void squarecpu(unsigned *matrix, unsigned *result, unsigned matrixsize) {
  for (unsigned ii = 0; ii < matrixsize; ii++) {
    for (unsigned jj = 0; jj < matrixsize; jj++) {
      for (unsigned kk = 0; kk < matrixsize; ++kk) {
        result[ii * matrixsize + jj] +=
            matrix[ii * matrixsize + kk] * matrix[kk * matrixsize + jj];
      }
    }
  }
}
int main() {
  unsigned matrix[4096];
  unsigned resultcpu[4096] = {0};
  unsigned resultgpu[4096] = {0};
  unsigned *dresultgpu, *dmatrix;
  for (int i = 0; i < 4096; i++) {
    matrix[i] = rand() % 10;
    // resultcpu[i] = 0;
    // resultgpu[i] = 0;
  }
  unsigned matrixsize = 64;

  unsigned start = clock();
  squarecpu(matrix, resultcpu, matrixsize);
  unsigned end = clock();
  double cpu_time_used = (((double)(end - start)) / CLOCKS_PER_SEC) * 1000;
  printf("time taken by cpu is %f ms\n", cpu_time_used);

  unsigned startgpu = clock();
  cudaSetDevice(1);
  cudaMalloc(&dresultgpu, 4096 * sizeof(int));
  cudaMalloc(&dmatrix, 4096 * sizeof(int));
  cudaMemcpy(dresultgpu, resultgpu, 4096 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dmatrix, matrix, 4096 * sizeof(int), cudaMemcpyHostToDevice);
  squaregpu1<<<matrixsize, matrixsize>>>(dmatrix, dresultgpu);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  cudaMemcpy(resultgpu, dresultgpu, 4096 * sizeof(int), cudaMemcpyDeviceToHost);
  unsigned endgpu = clock();
  double gpu_time_used =
      (((double)(endgpu - startgpu)) / CLOCKS_PER_SEC) * 1000;
  printf("time taken by gpu is %f ms\n", gpu_time_used);

  // CHECK result
  for (int i = 0; i < 4096; i++) {
    if (resultcpu[i] != resultgpu[i]) {
      printf("Incorrect result at index %d: CPU %u, GPU %u\n", i, resultcpu[i],
             resultgpu[i]);
      break;
    }
  }
  return 0;
}
