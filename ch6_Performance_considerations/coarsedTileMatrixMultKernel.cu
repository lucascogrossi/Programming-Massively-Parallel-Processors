#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void coarsedTiledMatrixMultKernel(float *M, float *N, float *P, int width) {
    __shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;   int by = blockIdx.y;
    int tx = threadIdx.x;  int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int tile = 0; tile < width/TILE_WIDTH; tile++) {

        // Collaborative loading of M tile into shared memory
        M_s[ty][tx] = M[row * width + tile*TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            N_s[ty][tx] = N[(tile*TILE_WIDTH + ty) * width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; k++) {
                Pvalue[c] += M_s[ty][k] * N_s[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }

}

void tiledMatrixMult(float *M_h, float *N_h, float *P_h, int width) {
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
    dim3 numBlocks((width + numThreadsPerBlock.x * COARSE_FACTOR - 1) / (numThreadsPerBlock.x * COARSE_FACTOR),
                   (width + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y
    );
    coarsedTiledMatrixMultKernel<<< numBlocks, numThreadsPerBlock >>>(M_d, N_d, P_d, width);
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

    tiledMatrixMult(M, N, P, n);

    printf("P[0] = %f | Expected:", P[0]);
    float sum = 0;
    for (int k = 0; k < n; k++) {
        sum += M[k] * N[k * n];
    }
    printf("%f\n", sum);


    free(M);
    free(N);
    free(P);

    return 0;
}