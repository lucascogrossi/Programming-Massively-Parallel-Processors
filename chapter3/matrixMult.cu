#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrixMultKernel(float *M, float *N, float *P, int width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < width) {
        float Pvalue = 0;
        for (int k = 0; k < width; k++) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = Pvalue;
    }
}

void matrixMult(float *M_h, float *N_h, float *P_h, int width) {
    // Allocate GPU memory
    float *M_d, *N_d, *P_d;
    checkCuda( cudaMalloc((void**)&M_d, width * width * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&N_d, width * width * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&P_d, width * width * sizeof(float)) );

    // Transfer data host -> device
    checkCuda( cudaMemcpy(M_d, M_h, width * width * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(N_d, N_h, width * width * sizeof(float), cudaMemcpyHostToDevice) );

    // Perform mmult
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (width + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y
    );
    matrixMultKernel<<< numBlocks, numThreadsPerBlock >>>(M_d, N_d, P_d, width);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    // Transfer data device -> host
    checkCuda( cudaMemcpy(P_h, P_d, width * width * sizeof(float), cudaMemcpyDeviceToHost) );
    
    // Free GPU memory
    checkCuda( cudaFree(M_d) );
    checkCuda( cudaFree(N_d) );
    checkCuda( cudaFree(P_d) );
}

int main(void) {
    srand(time(NULL));

    // N x N matrix
    int n = 1 << 10;

    float *M = (float*) malloc(n * n * sizeof(float));
    float *N = (float*) malloc(n * n * sizeof(float));
    float *P = (float*) malloc(n * n * sizeof(float));

    if(!M || !N || !P) {
        fprintf(stderr, "Malloc Error.\n");
        return 1;
    }

    for (int i = 0; i < n * n; i++) {
        M[i] = rand() / (float)RAND_MAX;
        N[i] = rand() / (float)RAND_MAX;
    }

    matrixMult(M, N, P, n);

    printf("P[0] = %f | Expected:", P[0]);
    float sum = 0;
    for (int k = 0; k < width; k++) {
        sum += M[k] * N[k * width];
    }
    P_verify[0] = sum;
    printf("%f\n", P_verify[0]);


    free(M);
    free(N);
    free(P);

    return 0;
}