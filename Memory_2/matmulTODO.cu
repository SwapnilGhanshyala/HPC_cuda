#include <cuda.h>
#include <stdio.h>
#include <time.h>

__global__ void CoalesingAwareSquaregpu(unsigned *matrix, unsigned *result) {
  unsigned ii = blockDim.x * blockIdx.x;
  unsigned k = threadIdx.x;
  unsigned matrixsize = blockDim.x;
  for (unsigned j = 0; j < matrixsize; j++)
    // result[ii + j] += matrix[ii + k] * matrix[k * blockDim.x + j];
    atomicAdd(&result[ii + j], matrix[ii + k] * matrix[k * matrixsize + j]);
}

__global__ void squaregpu(unsigned *matrix, unsigned *result) {
  unsigned ii = blockDim.x * blockIdx.x;
  unsigned j = threadIdx.x;
  unsigned matrixsize = blockDim.x;
  for (unsigned k = 0; k < matrixsize; k++)
    result[ii + j] += matrix[ii + k] * matrix[k * blockDim.x + j];
}
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

void cacheAwareSquarecpu(unsigned *matrix, unsigned *result,
                         unsigned matrixsize) {
  for (unsigned ii = 0; ii < matrixsize; ii++) {
    for (unsigned kk = 0; kk < matrixsize; ++kk) {
      for (unsigned jj = 0; jj < matrixsize; jj++) {
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
  }
  unsigned matrixsize = 64;

  unsigned start = clock();
  // squarecpu(matrix, resultcpu, matrixsize);
  cacheAwareSquarecpu(matrix, resultcpu, matrixsize);
  unsigned end = clock();
  double cpu_time_used = (((double)(end - start)) / CLOCKS_PER_SEC) * 1000;
  printf("time taken by cpu is %f ms\n", cpu_time_used);

  cudaSetDevice(1);
  cudaMalloc(&dresultgpu, 4096 * sizeof(int));
  cudaMalloc(&dmatrix, 4096 * sizeof(int));
  unsigned startgpu = clock();
  cudaMemcpy(dresultgpu, resultgpu, 4096 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dmatrix, matrix, 4096 * sizeof(int), cudaMemcpyHostToDevice);
  // squaregpu<<<matrixsize, matrixsize>>>(dmatrix, dresultgpu);
  CoalesingAwareSquaregpu<<<matrixsize, matrixsize>>>(dmatrix, dresultgpu);
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