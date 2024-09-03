// M is 1024 x 1024
// each Tblock works on 1024 elements that is 1 row

#include <stdio.h>
__global__ void replaceNoSharedMemory(unsigned *M) {
  unsigned jj = threadIdx.x;
  if (jj < 1023) {
    unsigned ii = blockIdx.x * 1024;
    // M[ii + jj] = M[ii + jj] + M[ii + jj + 1];
    atomicAdd(&M[ii + jj], M[ii + jj] + M[ii + jj + 1]);
  }
}
__global__ void replaceWithSharedMemory(unsigned *M) {
  // STEP 1: copy to shared memory
  // STEP 2: update the shared memory
  // STEP 3: copy back to global memory
  // even in this case a barrier/ or synchronization of some sort is needed.
  // to make sure that all the threads have completed the call.
  // to avoid using a barrier here, in step 2 , read from global memory and
  // write to shared memory.
  // but still step 2 and step 3 need a barrier because other warps might be
  // reading from global memory (step 2) when this warp is writing to global
  // memory(step 3).

  __shared__ unsigned nmij;
  unsigned jj = threadIdx.x;
  if (jj < 1023) {
    unsigned ii = blockIdx.x * 1024;
    nmij = M[ii + jj] + M[ii + jj + 1];
    __syncthreads();
    M[ii + jj] = nmij;
  }
}

int main() {
  const int size = 1024;
  const int bytes = size * sizeof(unsigned);

  // Allocate host memory
  unsigned h_input[size], h_output[size];

  // Initialize input data
  for (int i = 0; i < size; ++i) {
    h_input[i] = static_cast<float>(i);
  }

  // Allocate device memory
  unsigned *d_input;
  cudaMalloc(&d_input, bytes);

  // Copy input data to device
  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

  // Define block size and grid size
  const int blockSize = 256;
  const int gridSize = (size + blockSize - 1) / blockSize;

  // Launch kernel with dynamic shared memory allocation
  replaceWithSharedMemory<<<1024, 1024>>>(d_input);

  // Copy output data back to host
  cudaMemcpy(h_output, d_input, bytes, cudaMemcpyDeviceToHost);

  // Print some output data
  for (int i = 0; i < 10; ++i) {
    std::cout << h_output[i] << " ";
  }
  std::cout << std::endl;

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}