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









