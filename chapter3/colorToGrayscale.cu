__global__ void colorToGrayscaleKernel(unsigned char *Pin_d, unsigned char *Pout_d, int width, int height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        
        unsigned char r = Pin_d[rgbOffset];
        unsigned char g = Pin_d[rgbOffset + 1];
        unsigned char b = Pin_d[rgbOffset + 2];
        
        Pout_d[grayOffset] = (unsigned char)(0.2126f*r + 0.7152f*g + 0.0722f*b);
    }
}