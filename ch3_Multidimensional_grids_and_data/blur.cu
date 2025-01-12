__global__ void blurKernel(unsigned char *in, unsigned char *out, int width, int height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    pixVal += in[curRow*width + curCol];
                    pixels++:
                }

            }
            out[row * width + col] = (unsigned char) (pixVal/pixels);
        }
    }
}
