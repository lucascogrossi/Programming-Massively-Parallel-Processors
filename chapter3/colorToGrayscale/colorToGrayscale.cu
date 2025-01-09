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

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if(image.empty()) {
        std::cout << "Error: Could not read image" << std::endl;
        return -1;
    }

    cv::cuda::GpuMat gpu_color, gpu_gray;
    
    gpu_color.create(image.rows, image.cols, CV_8UC3); // RGB - 3 channels
    gpu_gray.create(image.rows, image.cols, CV_8UC1);  // Grayscale - 1 channel
    gpu_color.upload(image);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((image.cols + dimBlock.x - 1) / dimBlock.x,
                 (image.rows + dimBlock.y - 1) / dimBlock.y);

    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(
        gpu_color.ptr<unsigned char>(),
        gpu_gray.ptr<unsigned char>(),
        image.cols, image.rows
    );

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    cv::Mat gray_image;
    gpu_gray.download(gray_image);

    cv::imshow("Color", image);
    cv::imshow("Grayscale", gray_image);
    cv::waitKey(0);
    cv::imwrite("output.jpg", gray_image);

    return 0;
}