1. Consider the following CUDA kernel and the corresponding host function that calls it:

```
__global__ void foo_kernel(int *a, int *b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 2; // Line 04
    }
    if (i%2 == 0) {
        a[i] = b[i]*2; // Line 07
    }
    for (unsigned int j = 0; j < 5 - (i%3); ++j) { // Line 09
        b[i] += j;
    }
}

void foo(int *a_d, int *b_d) {
    unsigned int N = 1024;
    foo_kernel<<< (N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

a. What is the number of warps per block?

Each block has 128 threads -> 4 warps per block

b. What is the number of warps in the grid?

4 warps per block and we have 8 blocks in the grid -> 32 warps in the grid

c. For the statement on line 04:
i. How many warps in the grid are active?
For each block:  warp 0 is active, warp 1 is active but not all threads execute, warp 2 is inactive and warp 3 is active but not all threads execute. (3 warps)
We have 8 blocks in the grid -> 24 active warps.

ii. How many warps in the grid are divergent?

2 warps per block are divergent -> 2 * 8 = 16 divergent warps in the grid

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

warp 0(t0 - t31) all threads execute -> 100% SIMD efficiency.

iv. What is the SIMD efficiency (in %) of warp 1 of block 0?

warp 1(t32 - t63) only t32 - t39 execute  (8 threads total) -> 8/32 = 25% SIMD efficiency.

v. What is the SIMD efficiency (in %) of warp 3 of block 0?

warp 3(t96 - t127) only t104 - t127 execute (24 threads) -> 24/32 = 75% SIMD efficiency.

d. For the statement on line 07:

i. How many warps in the grid are active?

All 4 warps are active in a block -> All 32 warps are active in the grid.

ii. How many warps in the grid are divergent?

All 32 warps in the grid are divergent.

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

Only threads (t0, t2, t4 ... t30) are active (16 threads) -> 16/32 = 50% SIMD efficiency.

e. For the loop on line 09:

i. How many iterations have no divergence?
i % 3 can be 0, 1, or 2 -> 5, 4, 3 iterations respectively
The first 3 (j = 0, j = 1, j = 2) iterations have no divergence because all threads are still in the loop. After that (threads with i%3 = 2) have finished but the others continue, causing divergence. 

ii. How many iterations have divergence?

2 iterations have divergence.

2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

The grid will be launched with 4 blocks with 512 per block -> 2048 threads total

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

2048 threads are launched and 48 threads won't execute. Last warp won't execute at all so we are left with 2016 threads. 2000 threads will execute and 16 threads will not.

Only one warp will have divergence

4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads' total execution time is spent waiting for the barrier?

t0: will wait 3.0 - 2.0 = 1.0 ms -> 1.0 / 3.0 = 33.33%
t1: will wait 3.0 - 2.3 = 0.7 ms -> 0.7 / 3.0 = 23.33&
t2: will wait 3.0 - 3.0 = 0.0 ms -> 0.0 / 3.0 = 0%
t3: will wait 3.0 - 2.8 = 0.2 ms -> 0.2 / 3.0 = 6.67%
t4: will wait 3.0 - 2.4 = 0.6 ms -> 0.6 / 3.0 = 20%
t5: will wait 3.0 - 1.9 = 1.1 ms -> 1.1 / 3.0 = 36.67%
t6: will wait 3.0 - 2.6 = 0.4 ms -> 0.4 / 3.0 = 13.33%
t7: will wait 3.0 - 2.9 = 0.1 ms -> 0.1 / 3.0 = 3.33%

5. If a CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

It is not a good idea. While it might work in current architectures, future GPUs might have different warp sizes. Also it might be harder to debug the kernel if the number of threads in each block is changed in the future. Just leave the __syncthreads() instruction.

6. If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM.

(c) 512 threads per block - but we should use 3 full blocks for full occupancy in this SM.

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.

a. 8 blocks with 128 threads each. -> Possible (50% occupancy)
b. 16 blocks with 64 threads each -> Possible (50% occupancy)
c. 32 blocks with 32 threads each -> Possible (50% occupancy)
d. 64 blocks with 32 threads each -> Possible (100% occupancy)
e. 32 blocks with 64 threads each -> Possible (100% occupancy)

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64k (65536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

a. The kernel uses 128 threads per block and 30 registers per thread.

128 tpb would mean 16 blocks = total of 2048 threads per SM. Would use 61440 registers. It can achieve full occupancy.

b. The kernel uses 32 threads per block and 29 registers per thread.

32 tpb would mean 64 blocks and our SM is limited to 32 blocks. Would not be possible to achieve full occupancy. Limiting factor is the maximum number of blocks per SM.

c. The kernel uses 256 threads per block and 34 registers per thread.

256 tpb would mean 8 blocks = 2048 threads total per SM which means achieve full occupancy. However, it would not be possible because since each thread uses 34 registers, needing a total of 69632 and we only have 65536 registers per SM.

9. A student mentions that they were able to multiply two 1024x1024 matrices using a matrix multiplication kernel with 32x32 thread blocks. The student is using a CUDA device that allow up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

I would be very confused because his hardware limit is 512 threads per block. 

He is using 1024 (32x32) threads per block and would need 1024 blocks.

The kernel wouldn't even launch.












