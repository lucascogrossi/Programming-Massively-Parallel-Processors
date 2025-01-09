#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void vecAddKernel(float *A_d, float *B_d, float *C_d, unsigned int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        C_d[i] = A_d[i] + B_d[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, unsigned int n) {
    // Allocate GPU memory
    float *A_d, *B_d, *C_d;
    
    checkCuda( cudaMalloc((void**)&A_d, n * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&B_d, n * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&C_d, n * sizeof(float)) );

    // Transfer data host -> device
    checkCuda( cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice) );

    // Perform computation
    int numThreadsPerBlock = 512;
    int numBlocks = (n + numThreadsPerBlock  - 1) / numThreadsPerBlock;
    vecAddKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, n);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    // Transfer data device -> host
    checkCuda( cudaMemcpy(C_h, C_d, n * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free GPU memory
    checkCuda( cudaFree(A_d) );
    checkCuda( cudaFree(B_d) );
    checkCuda( cudaFree(C_d) );
}

int main(void) {
    unsigned int N = 1 << 20; // ~1mil elements

    float *A = (float*) malloc(N * sizeof(float));
    float *B = (float*) malloc(N * sizeof(float));
    float *C = (float*) malloc(N * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Malloc error.\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        A[i] = rand();
        B[i] = rand();
    }

    vecAdd(A, B, C, N);

    // Verify result
    printf("C[0] = %f | Expected: %f\n", C[0], A[0] + B[0]);

    free(A);
    free(B);
    free(C);
    
    return 0;
}