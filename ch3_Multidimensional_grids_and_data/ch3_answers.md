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
__global__ void mmult(float *M, float *N, float *P, int n) {
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

