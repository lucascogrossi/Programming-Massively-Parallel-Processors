#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

#define CHANNELS 3

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__ void colorToGrayscaleKernel(unsigned char *Pin_d, unsigned char *Pout_d, int width, int height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row * width + col;

        // One can think of the RGB image having CHANELL
        // times more columns than the grayscale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin_d[rgbOffset    ]; // Red value
        unsigned char g = Pin_d[rgbOffset + 1]; // Green value
        unsigned char b = Pin_d[rgbOffset + 2]; // Blue value

        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout_d[grayOffset] = (unsigned char) 0.2126f*r + 0.7152f*g + 0.0722f*b;

    }
}

// Allocation and transfer already done by openCV
void colorToGrayscale(unsigned char *Pin_d, unsigned char *Pout_d, int width, int height) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
                 (height + dimBlock.y - 1) / dimBlock.y);
    colorToGrayscaleKernel<<< dimGrid, dimBlock>>>(Pin_d, Pout_d, width, height);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
}

int main() {
    // Read image using OpenCV
    cv::Mat image = cv::imread("input.jpg");
    if(image.empty()) {
        std::cout << "Error: Could not read image" << std::endl;
        return -1;
    }

    // Create GPU mat
    cv::cuda::GpuMat gpu_color, gpu_gray;
    gpu_color.upload(image);

    // Convert to grayscale
    colorToGrayscale(gpu_color.ptr<unsigned char>(), gpu_gray.ptr<unsigned char>(), 
                     image.cols, image.rows);

    // Get result back to CPU
    cv::Mat gray_image;
    gpu_gray.download(gray_image);

    // Show images
    cv::imshow("Color", image);
    cv::imshow("Grayscale", gray_image);
    cv::waitKey(0);

    // Save result
    cv::imwrite("output.jpg", gray_image);

    return 0;
}