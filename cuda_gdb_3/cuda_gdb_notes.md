Debugging and Profiling

Debugging:
1) is difficult because:
    i) non-determinism due to thread scheduling : 
        a) determinitic: every run produces same output
        b) non-determinism: e.g. random number generator
        c) in parallel setting : printfs, writing to a file, might change if you run the program multiple times.
        d) arguing about correctness(correct result) is difficult when output is changing.
    ii) Output can be different.
    iii) intermediate values might be different on every run, even though the output is the same.
2) cuda-gdb: 
    i) extension to gdb: most of the gdb commands also work e.g., breakpoints, single step, read/write memory contents.
    ii) this is for real hardware
    iii) some issues can occur , like and a.out might generate error but not with gdb, e.g. single stepping through program might cause some timing issues which is different from running the program arbitrarily.
    iv) cuda errors : 0-no error/ Success, 700 : IllegalAddress
    v) nvcc -g -G cw1.cu
        a) -g : include symbol information in both gdb and cuda gdb
        b) -G : include kernel or symbol information of Kernel.
        c) both are requires
        d) caveat: some optimizations by the compiler get disabled because of these options.
        e) why remove optimizations? compiler can remove certain statements if not needed. when debugging , we expect that statement by stmt execution will occur.
    vi) cuda-gdb a.out
        a) run
        b) cuda-gdb works with a focus thread.
        c) LWP: light weight process.

        d)(cuda-gdb) info cuda kernels
            Kernel Parent Dev Grid Status                           SMs Mask GridDim BlockDim Invocation 
            *      0      -   0    1 Active 0x00000000000000000000000000000005 (2,1,1) (10,1,1) K()   
        e)(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count         Virtual PC Filename  Line 
Kernel 0
*  (0,0,0)   (0,0,0)     (1,0,0)      (9,0,0)    20 0x00007fffe2e0b430              0 
        f)(cuda-gdb) cuda block 1 thread 0
[Switching focus to CUDA kernel 0, grid 1, block (1,0,0), thread (0,0,0), device 0, sm 2, warp 0, lane 0]
8       }

        g) breaks:
            --> break main
            --> break file.cu:233 // file line
        h) enter : skip step
        i) info cuda lanes.