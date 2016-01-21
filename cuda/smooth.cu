
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
const unsigned int FILTER_SIZE = 25;

// function to check if there are any cuda errors
void cudaErrorCheck(cudaError_t error){
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda Error: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

// Device code
__global__ void smooth_kernel(const int width, const int height, const int size, const int spectrum, float * filter, unsigned char * inputImage, unsigned char * smoothImage) {
	int gridStride = blockDim.x * gridDim.x;

	for (int pos = blockIdx.x * blockDim.x + threadIdx.x; 
        pos < size; 
        pos += gridStride) {
		darkGrayImage[pos] = ((0.3f * (float)inputImageR[pos]) + (0.59f * (float)inputImageG[pos]) + (0.11f * (float)inputImageB[pos])) * 0.6f + 0.5f;
    }
    /////////////////

    for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			for ( int z = 0; z < spectrum; z++ ) {
				unsigned int filterItem = 0;
				float filterSum = 0.0f;
				float smoothPix = 0.0f;

				for ( int fy = y - 2; fy < y + 3; fy++ ) {
					if ( fy < 0 ) {
						filterItem += 5;
						continue;
					}
					else if ( fy == height ) {
						break;
					}
					
					for ( int fx = x - 2; fx < x + 3; fx++ ) {
						if ( (fx < 0) || (fx >= width) ) {
							filterItem++;
							continue;
						}

						smoothPix += static_cast< float >(inputImage[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
						filterSum += filter[filterItem];
						filterItem++;
					}
				}

				smoothPix /= filterSum;
				smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix + 0.5f);
			}
		}
	}
}

// Host code
void triangularSmooth(const int width, const int height, const int spectrum, unsigned char * inputImage, unsigned char * smoothImage) {
	// initialize timers
	NSTimer kernelTime = NSTimer("kernelDarker", false, false);	
	NSTimer allocationTime = NSTimer("allocationDarker", false, false);
	NSTimer initTime = NSTimer("initDarker", false, false);
	NSTimer copyDeviceTime = NSTimer("copyDeviceDarker", false, false);
	NSTimer copyHostTime = NSTimer("copyHostDarker", false, false);
	NSTimer freeTime = NSTimer("freeDarker", false, false);
	// init vars
	cudaError_t error = cudaSuccess;
	unsigned char *inputImageDevice, *smoothImageDevice;
	float * filterDevice;
	int sizeImage = width * height * 3;
	// init call to setup cuda
	initTime.start();
	cudaSetDevice(0);
	initTime.stop();

	// allocate images in device memory
	allocationTime.start();
	error = cudaMalloc(&inputImageDevice, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	error = cudaMalloc(&smoothImageDevice, sizeImage * sizeof(unsigned char));
	cudaErrorCheck(error);
	error = cudaMalloc(&filterDevice, FILTER_SIZE * sizeof(float));
	cudaErrorCheck(error);
	allocationTime.stop();

	// Copy image from host to device
	copyDeviceTime.start();
	error = cudaMemcpy(inputImageDevice, inputImage, sizeImage, cudaMemcpyHostToDevice);
	cudaErrorCheck(error);
	error = cudaMemcpy(filterDevice, filter, FILTER_SIZE*sizeof(float), cudaMemcpyHostToDevice);
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
	smooth_kernel<<<numBlocks, threadsPerBlock>>>(width, height, sizeImage, spectrum, filterDevice, inputImageDevice, smoothImageDevice);
	cudaErrorCheck(cudaGetLastError());
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the result from device to host
	copyHostTime.start();
	error = cudaMemcpy(smoothImage, smoothImageDevice, sizeImage, cudaMemcpyDeviceToHost);
	cudaErrorCheck(error);
	copyHostTime.stop();
	
	// Free the images in the device memory
	freeTime.start();
	cudaFree(inputImageDevice);
	cudaFree(filterDevice);
	cudaFree(smoothImageDevice);
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
