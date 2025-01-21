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

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

```
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float *A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?

The kernel will only execute correctly when BLOCK_SIZE = 1 because the code is missing __syncthreads() when writing to shared memory.

b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

The root cause is lack of synchronization. Fix would be adding __syncthreads() after blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

11. Consider the following CUDA kernel and the corresponding host function that calls it:

```
__global__ void foo_kernel(float *a, float *b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];

    for (unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if (threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7*x[1] + 6.3f*x[2] + 8.5f*x[3]
           +y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}

void foo(int *a_d, int *b_d) {
    unsigned int N = 1024;
    foo_kernel<<< (N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

a. How many versions of the variable i are there?
1024.

b. How many versions of the array x[] are there?
1024.

c. How many versions of the variable y_s are there?
8.

d. How many versions of the array b_s[] are there?
8.

e. What is the amount of shared memory used per block (in bytes)?
516 bytes (128*4+4)

f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?

FLOPS = 10
Global memory accesses = 4 + 1 + 1 = 6 access => 24 bytes.

Answ: 10/24 = 0.42 OP/B

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65536) registers/SM and 96 KB of shared memory/SM. For each of the following kernel characteristics specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

a. The kernel uses 64 threads/block, 27 registers/thread, and 4KB of shared memory/block.

With 64 tpb, this would give us 32 blocks => 2048 threads running in the SM (good). Each thread uses 27 registers, so total registers usage: 55296 (within limits, good). Each block uses 4kb of smem, we have 32 blocks... so smem usage would be 128kb, which would exceed the limit we have. The kernel would not achieve full occupancy and the limiting factor is the smem size.

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/block

256 tbp would give us 8 blocks => 2048 threads running in the SM (good). Each thread uses 31 registers, so total register use: 63488(within limits, good). Each block uses 8kb of smem and we have 8 blocks in the grid... total smem usage: 64kb (within limits, good.) Therefore, the kernel can achieve full occupancy.

