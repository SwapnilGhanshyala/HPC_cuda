// there is a link list on the cpu, and it has a next pointer
// copy the ll from CPU to GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
struct Node {
  int id;
  int val;
  Node *next;
};
__global__ void passLL(Node *arr) {}

int main() {
  Node arr[10];
  Node *darr;
  cudaMalloc(&darr, 10 * sizeof(Node));
  cudaMemcpy(darr, arr, 10, cudaMemcpyHostToDevice);

  return 0;
}