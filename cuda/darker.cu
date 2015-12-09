
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

void cudaErrorCheck(cudaError_t error){
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

__global__ void darkGray_kernel(const int width, const int height, const int size, const unsigned char * inputImageR, const unsigned char * inputImageG, const unsigned char * inputImageB, unsigned char * darkGrayImage) {
	int stride = blockDim.x * gridDim.x;

	for (int pos = blockIdx.x * blockDim.x + threadIdx.x; 
         pos < size; 
         pos += stride) {
		darkGrayImage[pos] = ((0.3f * (float)inputImageR[pos]) + (0.59f * (float)inputImageG[pos]) + (0.11f * (float)inputImageB[pos])) * 0.6f + 0.5f;
    }
}

// Host code
void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("kernelDarker", false, false);	
	NSTimer allocationTime = NSTimer("allocationDarker", false, false);
	NSTimer initTime = NSTimer("initDarker", false, false);
	NSTimer copyDeviceTime = NSTimer("copyDeviceDarker", false, false);
	cudaError_t error = cudaSuccess;
	unsigned char *inputImageDeviceR,*inputImageDeviceG, *inputImageDeviceB, *darkGrayImageDevice;
	int sizeImage = width * height;

	// init call to setup cuda
	initTime.start();
	cudaSetDevice(1);
	initTime.stop();

	// allocate images in device memory
	allocationTime.start();
	error = cudaMalloc(&inputImageDeviceR, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	error = cudaMalloc(&inputImageDeviceG, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	error = cudaMalloc(&inputImageDeviceB, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	error = cudaMalloc(&darkGrayImageDevice, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	allocationTime.stop();

	// Copy image from host to device
	copyDeviceTime.start();
	error = cudaMemcpy(inputImageDeviceR, inputImage, sizeImage, cudaMemcpyHostToDevice);
	error = cudaMemcpy(inputImageDeviceG, inputImage+sizeImage, sizeImage, cudaMemcpyHostToDevice);
	error = cudaMemcpy(inputImageDeviceB, inputImage+(sizeImage*2), sizeImage, cudaMemcpyHostToDevice);
	cudaErrorCheck(error);
	copyDeviceTime.stop();

	kernelTime.start();
	//dim3 threadsPerBlock(16, 16);
	//dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	int threadsPerBlock = 265;
	//must be a multiple of num SM's for optimal performance
	int numBlocks = 1024;
	darkGray_kernel<<<numBlocks, threadsPerBlock>>>(width, height, sizeImage inputImageDeviceR, inputImageDeviceG, inputImageDeviceB, darkGrayImageDevice);
	cudaErrorCheck(error);
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the result from device to host
	error = cudaMemcpy(darkGrayImage, darkGrayImageDevice, sizeImage, cudaMemcpyDeviceToHost);
	cudaErrorCheck(error);
	
	// Free the images in the device memory
	cudaFree(inputImageDeviceR);
	cudaFree(inputImageDeviceG);
	cudaFree(inputImageDeviceB);
	cudaFree(darkGrayImageDevice);
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << "initalization time: " << initTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "allocation time: " << allocationTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "kernel time:" << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() << endl;
}
