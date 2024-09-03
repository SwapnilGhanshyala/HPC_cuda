TAKEAWAYS: 
GPU are designed for wider data bus rather than fast data buses.

Techniques to improve bandwidth:
1) Share / re-use of data :
    i) is a threads bring in some data , let others also use it
    ii) if data is brought by a thread, if others need the data , let them re-use it
    iii) thsi requires thread to thread communication?
2) data compression (alternatively can we work on compressed data itself)
3) Recompute that store + fetch. 

latency:
1) overhead because of architecture's communication, memory i/o becomes the bottleneck in memory bound application.
2) Improvement:
    i) caches help
    ii) on GPU latency hiding is done by exploiting massive multithreading (which is cache efficient). Basically if a warp is waiting for I/O, execute another warp and on and on , untill the first warp can be taken over again. Large number of warps are available for multithreading.
3) latency hiding:

Locality heuristics:
1) Spatial locality: nearby data might get accessed together or in short span of time , eg a[i] is accessed so a[i+k] might also get accessed in the same loop.
2) Temporal locality: chance of re-use is high.

Memory Coalescing:
1) if all threads in a warp are accessing consequtive memory then all these fetchesa re clubbed into 1, this is coalesced.
2) Coalescing is hardware dependent, that is GPUs handle memory coalescing.
3) degree of Coalescing = 33-(the number of memory transactions required for a warp t execute an instruction)
4) we want high degree of coalescing.
    i) a[threadIdx.x] has DOC of 32 i.e, 33-1 (consequtive accessess by all threads)
5) CPU works well this array of structures AOS,
    i) threads should access consecutive elements of a chunk.
    ii) better locality on CPU.
    iii) When a thread accesses an atrribute of a node, then it also accesses the other attributes of the same node.
6) GPU works well with Structure of Array SOA,
    i) consequtive data item should be accessed by the next thread,
    ii) chunk should be accessed by consequtive threads.
    iii) Better coalescing on GPU.

Coherence Problem:
1) Race condition.
2) Different versions (or copies or views) of of data available on different memories availabe to different threads.
3) Hardware has to give certain guarantees about the transfer of data. example 1) scalar being read from a structure being written to.
4) exact GPU and CPU (generally known) protocals are different. 

Data Race
1) Occurs due to simultaneous accessess to a memory location and atleast 1 of them is a write.
Global Memory : 
1) GPU RAM/Video Memory/ VDU memory(older days)(also Graphics DRAM, version might be different from CPU (ddr3 on cpu and gddr on GPU))
2) shared by all threads(per GPU)
3) long latency (400-800 cycles)
4) Throughput ~200 GBPs
5) cudamalloc

L2 cache: 
1) ~768 KB 
2) Shared among SMs
3) Advantage for fast atomics
4) this isn't part of programming like Shared memory which is part of L1 cache.

Shared Memory and L1 split:
// hardware's L1 cache can be split into 2 parts , l1 cache and Shared memory
1) Shared memory - every thread in a block sharesthis.(per thread block)(scratch pad /L1)
    i) Since it is on the L1 cache , therefore it is shared by All threads on the SM , therefore shared accross all threads in a TB.
    ii) managed by programmer
    iii) inside the kernel, below is how you allocate a shared memory:
        __shared__ float a[N];
        __shared__ unsigned s;
    iv) accessing :
        a[id]=id; // id has to be threadId, it cannot be the global ID, because shared memory is accessed in a Thread block
        // in above case if id is some global id, then all threads will be accessing some same location.
        if (id==0) s=1;
    v) __syncthreads()
        a) not very costly even though it is adding sequentiality to the code rather 1 Thread block, the other thread block will execute in parallel.
        b) if there is a data race on a shared memory, global memory will fail as global memory is accessed nby all the thread blocks not just 1, you must add syncthreads.
        c) syncthread can be implemented using a counter.

2) L1 cache :
    i) low latency 20-30 cycles 
    ii) High bandwidth (~1 TBPS)
    iii) available per SM
    iv) managed by hardware

3) cudaDeviceSetCacheConfig(kernelname, param);
    i) param is {cudaFuncCachePreferNone (default setting), or ...PreferL1 (if access pattern is very regular), or ...Shared(more prgrammer decison making for the kernel)}, in newer generation , ...preferEqual (split equally).
4) Dynamic shared memory
    i) when it is not know how much memory is required at the compile time.
    ii) syntax: it is the 3rd parameter to the kernel launch.
        a) in the host:
            dkernel<<<1,n,n*sizeof(int)>>>();// where n is coming from the input/args to program.
        b) in the kernel:
            extern __shared__ int s[]; // kernel now knows the size of this array is coming from 3rd parameter of kernel call.
    iii) best used when declaring arrays in the kernel.
    iv) can we allocate multiple dynamic shared memory specifies in the kernel call? 
        a) specify multiple arrays in the kernel and single large block in kernel call(i.e., host)
        b) in kernel : extern __shared__ int s[];
            int *s1 = s;
            int *s2 = s+n1; // can be of another type if you want multiple types
        c) for multiple data types
        -->NOTE : this will have padding issues
        so programmer has to take care of misallignments.

5) Bank Conflicts
    i) Shared memory is organized into banks.
    ii) multiple accesses to the same banks are served sequentially.

Registers : 
1) each thread has its own (per thread)
2) Not visible to programmer
3) ~8TBPS bandwidth
    i) local variables will go here 
    ii) if local vars spill into global memory then they will obviously be handled by L2 and L1 cache.
4) ~32K or 64 K in number per SM, there is 1 register file per SM
5) Compiler cannot always know how many registers are required because #threads is dynamic or variable and known at runtime.

Texture Memory:
1) shared by all threads(per GPU)
2) Read-only (not writable from kernel)
3) Size is ~12KB , ie, much much smaller than Global memory
4) fast memories , so small amount of data that is very often used.
5) GPU has a texture cache that caches the data in texture memory that is being accessed. (this mechanism is slightly different fron CPU caching)
6) Optimized for 2D spatial locality
7) syntax: 
    i) definition: texture<float,2,cudaReadModeElementType> tex;// <type, number of dimensions, read mode in kernel>
        -->global definition of texture so that both main and kernel know what it is refering to.
    ii) in main: cudaBindTextureToArray(tex,cuArray,...);// tells that cuArray is going to be accessed as a 2D matrix 
    iii) kernel: ...=tex2D(tex,...); // to access a particular element of the texture
        --> tex2D(tex2D(tex,tu+0.5,tv+0.5));// (texname, index x , index y);


Constant Memory
1) shared by all threads(per GPU)
2) ~64 KB Read-only (similar use case as the Tex Memory)
3) also has cache
4) Syntax:
    i) definition: __constant__ unsigned meta;
    -->global
    ii)main: cudamemcpyToSymbol(meta,&hmeta,sizeof(unsigned));// hmeta is copied to meta
    --> unsigned hmeta=10;
    iii) Kernel: data[threadIdx.x]=meta[0];

CUDA takeaways till now:
1) Compute unified device Architecture : it is hardware software architecture.
2) consistency and coherence has to be taken care of by cuda programmer
3) compiler and/or programmer can decide blocks and grids.
4) GPU (driver) instantiates a grid of this kernel and distributes it across all SM to do load balancing based on the hardware capability. which is why threadblocks are ideally, independent of each other.
5) Register spill is by default handled on global memory