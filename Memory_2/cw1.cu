// write a kernel to vary the degree of coalesing from 1 to 32 based on an input
// argument. basically n is the stride
__global__ void varyDOC(unsigned *a, int n) {
  a[threadIdx.x * n] // I think this works
                     // if n=1 then consequtive and DOC is 32
  // if n==2 then 32 threads will access alternate location over 64 memory
  // locations
}