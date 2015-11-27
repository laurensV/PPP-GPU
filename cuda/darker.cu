
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
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int pos = (y * width) + x;

	// Check if we are inside the bounds of the image
	if (x < width && y < height){
		float grayPix = 0.0f;
		float r = static_cast< float >(inputImage[pos]);
		float g = static_cast< float >(inputImage[size + pos]);
		float b = static_cast< float >(inputImage[(2 * size) + pos]);

		grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
		grayPix = (grayPix * 0.6f) + 0.5f;

		darkGrayImage[pos] = static_cast< unsigned char >(grayPix);
	}
}

// Host code
void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("kernelDarker", false, false);	
	NSTimer allocationTime = NSTimer("allocationDarker", false, false);
	size_t sizeInputImage = width * height * 3 * sizeof(unsigned char);
	size_t sizedarkGrayImage = width * height * sizeof(unsigned char);

	// allocate images in device memory
	unsigned char *inputImageDevice, *darkGrayImageDevice;

	allocationTime.start();
	cudaMalloc((void **) &inputImageDevice, sizeInputImage);
	cudaMalloc((void **) &darkGrayImageDevice, sizedarkGrayImage);
	allocationTime.stop();
	// Copy image from host to device
	cudaMemcpy(inputImage_device, inputImage, sizeInputImage, cudaMemcpyHostToDevice);

	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	darkGray_kernel<<<numBlocks, threadsPerBlock>>>(width, height, width*height, inputImage, darkGrayImage);
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the result from device to host
	cudaMemcpy(darkGrayImage, darkGrayImageDevice, sizedarkGrayImage, cudaMemcpyDeviceToHost);
	
	// Free the images in the device memory
	cudaFree(inputImage_device);
	cudaFree(grayImage_device);
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() << endl;
	cout << fixed << setprecision(6) << allocationTime.getElapsed() << setprecision(3) << endl;


}
