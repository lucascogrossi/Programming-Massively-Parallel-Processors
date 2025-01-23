#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define COARSE_FACTOR 4

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void tiledMatrixMultKernel(float *M, float *N, float *P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    __shared__ float M_s[TILE_DIM][TILE_DIM];
    __shared__ float N_s[TILE_DIM][TILE_DIM];

    float sum[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }

    for (int tile = 0; tile < width/TILE_DIM; tile++) {
        M_s[threadIdx.y][threadIdx.x] = M[row * width + tile*TILE_DIM + threadIdx.x];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c*TILE_DIM;

            N_s[threadIdx.y][threadIdx.x] = N[(tile*TILE_DIM + threadIdx.y) * width + col];
            __syncthreads();

            for (int i = 0; i < TILE_DIM; i++) {
                sum[c] += M_s[threadIdx.y][i] * N_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c*TILE_DIM;
        P[row * width + col] = sum[c];
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
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/COARSE_FACTOR / (numThreadsPerBlock.x ),
                   (width + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    tiledMatrixMultKernel<<< numBlocks, numThreadsPerBlock >>>(M_d, N_d, P_d, width);
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
    int n = 1 << 12;

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