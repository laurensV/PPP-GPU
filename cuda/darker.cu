
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Device code
__global__ void darkGray_kernel(const int width, const int height, const int size, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	// get position of thread in the 'thread matrix'
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Check if we are outside the bounds of the image
	if (x >= width || y >= height) return;

	int pos = (y * width) + x;

	float grayPix = 0.0f;
	float r = static_cast< float >(inputImage[pos]);
	float g = static_cast< float >(inputImage[size + pos]);
	float b = static_cast< float >(inputImage[(2 * size) + pos]);

	grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
	grayPix = (grayPix * 0.6f) + 0.5f;

	darkGrayImage[pos] = static_cast< unsigned char >(grayPix);
}

// Host code
void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("kernelDarker", false, false);	
	NSTimer allocationTime = NSTimer("allocationDarker", false, false);
	cudaError_t error = cudaSuccess;
	int sizeInputImage = width * height * 3;
	int sizedarkGrayImage = width * height;

	// allocate images in device memory
	unsigned char *inputImageDevice, *darkGrayImageDevice;

	allocationTime.start();
	error = cudaMalloc(&inputImageDevice, sizeInputImage * sizeof(unsigned char));
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error in cudaMalloc input image: %s\n", cudaGetErrorString(error));
		exit(1);
	}
	error = cudaMalloc(&darkGrayImageDevice, sizedarkGrayImage * sizeof(unsigned char));
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error in cudaMalloc gray image: %s\n", cudaGetErrorString(error));
		exit(1);
	}	
	allocationTime.stop();
	// Copy image from host to device
	error = cudaMemcpy(inputImageDevice, inputImage, sizeInputImage, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error in cudaMemcpy image from host to device: %s\n", cudaGetErrorString(error));
		exit(1);
	}	
	kernelTime.start();
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(sizeInputImage / threadsPerBlock.x, sizeInputImage / threadsPerBlock.y);
	darkGray_kernel<<<numBlocks, threadsPerBlock>>>(width, height, width*height, inputImageDevice, darkGrayImageDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error in darkGray_kernel: %s\n", cudaGetErrorString(error));
		exit(1);
	}		
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the result from device to host
	error = cudaMemcpy(darkGrayImage, darkGrayImageDevice, sizedarkGrayImage, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error in cudaMemcpy result from device to host: %s\n", cudaGetErrorString(error));
		exit(1);
	}		
	// Free the images in the device memory
	cudaFree(inputImageDevice);
	cudaFree(darkGrayImageDevice);
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() << endl;
	cout << fixed << setprecision(6) << allocationTime.getElapsed() << setprecision(3) << endl;
}
