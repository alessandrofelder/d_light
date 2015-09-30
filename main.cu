#include <stdio.h>
#include <tiffio.h>
#include <assert.h>
#include <iostream>
#include <limits>
#include <string>

#include <cuda.h>
#include <helper_cuda.h>

typedef unsigned short GreyscaleValue; //unsigned char for 8-bit and unsigned short for 16-bit tiff
typedef double Real;

#include "flatFieldCorrect_cpu.h"
#include "flatFieldCorrect_gpu.h"
#include "flatFieldCorrect_kernel.h"
#include "invert_kernel.h"
#include "fieldImages.h"

int main(int argc, const char **argv)
{

	 // initialise card
	findCudaDevice(argc, argv);

	//input data
	const char* lightFile = "../test-d_light/test-data/16-bit/MED_light.tif";
	const char* darkFile = "../test-d_light/test-data/16-bit/MED_dark.tif";
	const char* fileToCorrect = "../test-d_light/test-data/16-bit/lamb1_";
	int nImages = 1;

	float milli =0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	FieldImages fi(lightFile, darkFile);
	flatFieldCorrect_cpu<GreyscaleValue, Real>(fi, fileToCorrect,nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("testing execution of sequential flat field correction (16-bit): %.1f (ms)", milli);

	cudaEventRecord(start);
	flatFieldCorrect_gpu<GreyscaleValue, Real>(fi, fileToCorrect,nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("testing execution of parallel flat field correction (16-bit): %.1f (ms)", milli);


	cudaDeviceReset();
    return 0;

}
