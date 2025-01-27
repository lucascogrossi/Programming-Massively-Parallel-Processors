1. Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

[See cornerTurning.cu](cornerTurning.cu)

2. For tiled matrix multiplication, of the possible range of values for
BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)

It will avoid uncoalesced access to global memory when BLOCK_SIZE is a multiple of 32 (warp size).

3. Consider the following CUDA kernel:

```
__global__ void foo_kernel(float *a, float *b, float *c, float *d, float *e) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i]; // Line 05
    for (unsigned int j = 0; j < 4; j++) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i * 4 + j];
    }
    __syncthreads();
    d[i + 8] = a_s[threadIdx.x];
    e[i * 8] = bc_s[threadIdx.x * 4];

}
```

For each of the following memory accesses, specify whether they are
coalesced or uncoalesced or coalescing is not applicable:

a. The access to array a of line 05
Coalesced.

b. The access to array a_s of line 05
Not applicable - smem.

c. The access to array b of line 07
Coalesced.

d. The access to array c of line 07
Uncoalesced.

e. The access to array bc_s of line 07
Not applicable - smem.

f. The access to array a_s of line 10
Not applicable - smem.

g. The access to array d of line 10
Coalesced.

h. The access to array bc_s of line 11
Not applicable - smem.

i. The access to array e of line 11
Uncoalesced.

3.
TODO