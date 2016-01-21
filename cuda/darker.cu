
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// function to check if there are any cuda errors
void cudaErrorCheck(cudaError_t error){
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

// Device code
__global__ void darkGray_kernel(const int width, const int height, const int size, const unsigned char * inputImageR, const unsigned char * inputImageG, const unsigned char * inputImageB, unsigned char * darkGrayImage) {
	int gridStride = blockDim.x * gridDim.x;

	for (int pos = blockIdx.x * blockDim.x + threadIdx.x; 
        pos < size; 
        pos += gridStride) {
		darkGrayImage[pos] = ((0.3f * (float)inputImageR[pos]) + (0.59f * (float)inputImageG[pos]) + (0.11f * (float)inputImageB[pos])) * 0.6f + 0.5f;
    }
}

// Host code
void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	// initialize timers
	NSTimer kernelTime = NSTimer("kernelDarker", false, false);	
	NSTimer allocationTime = NSTimer("allocationDarker", false, false);
	NSTimer initTime = NSTimer("initDarker", false, false);
	NSTimer copyDeviceTime = NSTimer("copyDeviceDarker", false, false);
	NSTimer copyHostTime = NSTimer("copyHostDarker", false, false);
	NSTimer freeTime = NSTimer("freeDarker", false, false);
	// init vars
	cudaError_t error = cudaSuccess;
	unsigned char *inputImageDeviceR,*inputImageDeviceG, *inputImageDeviceB, *darkGrayImageDevice;
	int sizeImage = width * height;
	// init call to setup cuda
	initTime.start();
	cudaSetDevice(0);
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
	cudaErrorCheck(error);
	error = cudaMemcpy(inputImageDeviceG, inputImage+sizeImage, sizeImage, cudaMemcpyHostToDevice);
	cudaErrorCheck(error);
	error = cudaMemcpy(inputImageDeviceB, inputImage+(sizeImage*2), sizeImage, cudaMemcpyHostToDevice);
	cudaErrorCheck(error);
	copyDeviceTime.stop();

	// number of SM's for GeForce GTX 480
	int numSMs = 32;
	// number of threads per block for GeForce GTX 480
	int threadsPerBlock = 1024;
	// must be a multiple of num SM's for optimal performance
	int numBlocks = 32*numSMs;

	// start the kernel
	kernelTime.start();
	darkGray_kernel<<<numBlocks, threadsPerBlock>>>(width, height, sizeImage, inputImageDeviceR, inputImageDeviceG, inputImageDeviceB, darkGrayImageDevice);
	cudaErrorCheck(cudaGetLastError());
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the result from device to host
	copyHostTime.start();
	error = cudaMemcpy(darkGrayImage, darkGrayImageDevice, sizeImage, cudaMemcpyDeviceToHost);
	cudaErrorCheck(error);
	copyHostTime.stop();
	
	// Free the images in the device memory
	freeTime.start();
	cudaFree(inputImageDeviceR);
	cudaFree(inputImageDeviceG);
	cudaFree(inputImageDeviceB);
	cudaFree(darkGrayImageDevice);
	freeTime.stop();

	// output times
	cout << fixed << setprecision(6) << "Initalization time: " << initTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "Allocation time: " << allocationTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "Copy to device time:" << copyDeviceTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "Kernel time:" << kernelTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "Copy to host time:" << copyHostTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "Free time:" << freeTime.getElapsed() << setprecision(3) << endl;
	cout << fixed << setprecision(6) << "GFLOP/s:" << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTime.getElapsed() << endl;
	cout << fixed << setprecision(6) << "GB/s:" << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() << endl;
}
