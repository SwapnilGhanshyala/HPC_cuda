Queries and lookups:
1) Time sharing or Time multiplexing on CPU and GPU
2) Hardware scheduling and driver's (software component's) role in it.
3) why does OS need scheduling and code translation and optimization if Compiler is already present in the pipeline?
4) Thread scheduling, Warp scheduling , block schduling.
5) "signal or Polling based mechanism for signalling the device if it is cudamemcpyAsync or cudaMemcpy". what is polling based mechanism?
6) How does integrated GPU handle global var ? for example a globally declared "const char*" being used in the kernel. In cuda, compiler gives an error (not a seg fault at runtime).

Takeaways:
1) Blocks might not be scheduled in order, Similarly warps might not be schduled in order. However I observe that threads within a warp (32 threads) are scheduled in order (id sequence).
2) Discrete gpus have separate memory than it's CPU. Integrated Gpus (usually by same manufacturer) will have common RAM between CPU and GPU.
3) Separate Memory: Programmers responsibility to maintain and synchronize copies of variable between gpu and cpu.
4) In cw3 changing N to 8000 from 1024 resulted in "Id 7999 is 0" instead. This is because the number of threads that can be specified to a gpu is limited (in this case to 1024). So to make use of for threads , we need to increase gridSize.
5) SM: has multiple cores in it, therefore can have multiple threads scheduled on it.
6) Multi-dimensional data has to be passed as 1-D , thus, the x,y,z dimensions become useful.
7) No concurrency in a single thread execution. problem when multiple threads trying to work on shared data. (Race condition, Reader-Writer problem, synchronization issue)
8) Graphics processor were non-programmable (some instructions fixed at hardware level). in early 2000's shading units were made programmable, showed increase in throughput.
9) GPU vendors: NVIDIA,AMD,INTEL,ARM,Qualcomm,Broadcom,Matrox Graphics, Vivante, Samsung
10) GPU languages: CUDA (Compute unified Device language, proprietary to NVIDA), OpenCL (Open Comuting language, open Source , on all computing devices(on cpus as well)), OpenAcc(Open Accelerator,universal to all accelerators(many core architecture) (technically can run on cpu as well, although not parallel)) 
11) Interfaces : Python -> Cuda, Javascript->OpenCL, LLVM->PTX. Might not give as good a performance as writing in the parallel language directly.
12) SM: Streaming Multiprocessors, multiple SM on GPU chip
13) Cuda cores : multiple cuda cores per SM
14) Tensor Cores
15) CPU clock freq 2.5~3 , GPU's server end freq is usually around 0.7 GHz 
16) also in cw8 it seems cuda malloc is a huge contributor to the delay, otherwise the execution is definitely faster on GPU because it has enough work to be be done in parallel.
17) Thread Blocks (ceil of 1024 threads as of now):
    i) threads within a TB share a very fast cache called shared memory (CUDA) or local memory (OpenCL)
    ii) threads within a TB can synchronize very fast because of the shared memory/local memory or scrathpad memory.
    iii) A TB is assigned to 1 SM(Streaming multiprocessor) till the task is finished. While within an SM , warps (from same TB) can be issued to different cores.
    iv) multiple SMs form the GPU. Grid can span multiple of these SMs.
    v) Redident Blocks : how many blocks are resident on SM, this concept comes into play when talking about barriers.
18) Cudamemcpy
    i) does not involve kernel execution or threads on GPU , it is more like a DMA transfer
19) Warp Execution:
    i) Single instruction multiple Data (SIMD)
    ii) Degree of Divergence (DoD): number of steps required by a warp to finish 1 instruction.
        a) DoD = 1 , if no thread divergence
        b) Dod = 32, fully divergent
    iii) Thread divergence : threads in a warp
    iv) in a ten way branch, what is the DoD? is it max or sum of the construct?
    v) conditions are not bad, conditions evaluating to different truth-value is also not bad. But conditions evaluating to different truth value WITHIN A WARP is the problem.

Compute capability of cuda
1) Hardware Version: Major.Minor e.g., 6.2 
    i) if cuda is latest but gpu is older gen then specify -arch=sm_62
    ii) macro __CUDA_ARCH__ is defined only in the device or kernel code. its value is some gen number e.g., 620
    iii) Major number defines different families of gpu, e.g.,
        a) 1 is Tesla and 2 is fermi - atomics,warp-vote,__syncthreads
        b) 3: kepler - Unified Memory(3.0), 3.5(DynamicParallelism : at the language level, nested kernels, however amount of parallelism that can be exploited has some limitations)
        c) 5: Maxwell
        d)  6: pascal- Atomics double (prior only 4 byte data types were supported , and 8 bytes was simulated)
        e) 7: Volta-Tensor core (eg, matmul computations)
        f) 8 : Turing- Hardware async copy (earlier, it was library support)
        g)  latest 9 : Ampere (A100 etc) , 10 : lovelace , 11: hopper
    iv)

2) Cuda software version 10.1 etc.
