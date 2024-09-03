1) Write matmul code for CPU, locality aware matmul on CPU, matmul on GPU and again the locality aware CPU matmul to GPU.
--> coalescing aware gpu square in matmulTODO.cu, performs worst.
--> the problem is it needs atomics because there multiple threads work on the same result cell [ii,j].
2) can we allocate multiple dynamic shared memory specifies in the kernel call?
    a) specify multiple arrays in the kernel and single large block in kernel call(i.e., host)
3) is there any constency on the hardware only and cannot be handled on software level on cuda
4) can register splills be handled on Shared Memory?