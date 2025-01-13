1. In this chapter we implemented a matrix multiplication kernel that has each
thread produce one output matrix element. In this question, you will implement differente matrix-matrix multiplications kernels and compare them.

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design

```
__global__ 
void mmult(float *M, float *N, float *P, int n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < n) {
        for (int curCol = 0; curCol < n; curCol++) {
            float Pvalue = 0;
            for (int k = 0; k < n; k++) {
                Pvalue += M[row * n + k] * N[k * n + curCol];
            }
            P[row * n + curCol] = Pvalue;
        }
    }
}

/*
dim3 dimBlock(1, 256);
dim3 dimGrid(1, (n + dimBlock.y - 1) / dimBlock.y);
*/
```

b. Write a kernel that has each thread produce one output matrix column. Fillin the execution configuration parameters for the design.

```
__global__
void mmult(float *M, float *N, float *P, int n) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < n) {
        for (int curRow = 0; curRow < n; curRow++) {
            float Pvalue = 0;
            for (int k = 0; k < n; k++) {
                Pvalue += M[curRow * n + k] * N[k * n + col];
            }
            P[curRow * n + col] = Pvalue;
        }
    }
}

/*
dim3 dimBlock(256, 1);
dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
*/
```
c. Analyze the pros and cons of each of the two kernels designs.

Both kernels have a straightforward thread indexing - there's no need to think about higher dimensions. However, both kernels are very inneficient since they are reading multiple times from the same data (row/column) and both are not taking advantage of the GPU's full computational capacity.

3. Consider the following CUDA kernel and the corresponding host function that calls it:

```
__global__ void foo_kernel(float *a, float *b, unsigned int M, unsigned int N) {
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockIdx.y + threadIdx.x;
    if (row < M && col < N) {
        b[row * N + col] = a[row * N + col]/2.1f + 4.8f; // Line 05
    }
}

void foo(float *a_d, float *b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
    fooKernel<<< gd, bd >>>(a_d, b_d, M, N);
}
```
a. What is the number of threads per block?

16 threads in the x dimension and 32 threads in the y dimension (512 threads).

b. What is the number of threads in the grid?

61440 threads.

c. What is the number of blocks in the grid?

20 blocks in the x dimension and 6 blocks in the y dimension (120 total blocks)

d. What is the number of threads that execute the code on line 05?

45000 threads.

4. Consider 2D matrix with a width of 400 and height of 500. The matrix is stored as a one-dimenional array. Specify the array index of the matrix element at row 20 and column 10:

a. If the matrix is stored in row-major order;

row * width + col = 20 * 400 + 10 = 8010

b. If the matrix is stored in column-major order

col * height + row = 10 * 500 + 20 = 5020

5. Consider a 3D tensor with a width of 400, a height of 500 and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

TODO
After learning about tensors :)

