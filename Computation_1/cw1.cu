#include <stdio.h>
#include <cuda.h>
#define N 100
__global__ void cw1()
{
    printf("%d\n",threadIdx.x*threadIdx.x);
}
int main(){
    cw1<<<1,N>>>();
    cudaDeviceSynchronize();
    return 0;
}