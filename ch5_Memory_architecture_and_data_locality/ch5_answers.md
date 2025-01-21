1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwith consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Shared memory won't benefit matrix addition because there is no reuse of data. Assuming one thread is assigned to one output element, each element of the output matrix is simply calculated by adding the two corresponding elements from the input matrices.

2. 
TODO

3. What type of incorrect execution behavior can happen if one forgot to use one or both __syncthreads() in the kernel of fig 5.9?

The first __syncthreads() ensures that all threads have finished loading the tiles of M and N into M_s and N_s before any of them can move forward. If one forgot to add this barrier, some thread could try accessing an element that was not loaded yet. The second __syncthreads() ensures that all threads have finished using the M and N elements in shared memory before moving on to load the next tile. If the programmer forgot this barrier, a thread could load next tile's elements too early and corrupt the input value of other threads.

4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

 The most valuable reason would be to reduce memory bandwidth:  One thread can load from global memory to shared memory and other threads can access that same value from shared memory, thus reducing the total number of global memory accesses. When dealing with registers, each thread has its own set and cannot access other thread's registers.

 5. For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

 The global memory accesses are reduced by a factor of 32.

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

512000 versions. 

7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

1000 versions.

8. Consider performing a matrix multiplication of two input matrices with dimensions N x N. How many times is each element in the input matrices requested from global memory when:

a. There is no tiling?

When there is no tiling, each element is requested N times.

b. Tiles of size T x T are used?

When tiling, each element is requested N/T times.

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.

a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second

FLOPS per thread: 36 FLOPS
Global memory accesses per thread: 7 * 4 bytes = 28 bytes
Kernel Compute-to-memory ratio: 36 FLOPS/28 bytes = 1.29 FLOPS/byte.

Compute-to-memory ratio of the device: 200GFLOPS/100GB/s = 2 FLOPS/byte.

The kernel ratio is less than the device's ratio. This means the device cannot provide enough memory bandwidth relative to compute power. Therefore, memory-bound. 

b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250GB/second.

 Kernel Compute-to-memory ratio is the same: 1.29 FLOPS/byte.

 Compyte-to-memory ratio of the device: 300GFLOPS/250GB/s = 1.2 FLOPS/byte.

The kernel ratio is greater than the device's ratio, compute-bound. This means the device cannot provide enough compute power relative to memory bandwidth. Thefore, compute-bound.

