#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS)*2 + 1)

// Allocate constant memory on the GPU
// Use cudaMemcpyToSymbol() in the host
__constant __ float mask_c[MASK_DIM][MASK_DIM];


// Parallelization approach: 
// Assign one thread to compute each output element 
// by iterating over input elements and mask weights
__global__ void convolution_kernel(float *input, float *output, unsigned int width, unsigned int height) {

    // Output thread index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (outRow < height && outCol < width) {

        float sum = 0.0f;
        
        // Loop through the mask
        for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
            for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                
                // Find what elements to work on the input
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;

                if((inRow < height && inRow >= 0) && (inCol < width && inCol >= 0)) {
                    sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
}
