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
#include "flatFieldCorrect_kernel.h"
#include "invert_kernel.h"

const int gridsize = 8*1024;
const int blocksize = 32;


int main(int argc, const char **argv)
{

	 // initialise card
	findCudaDevice(argc, argv);

	// initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//sequential reference version
	cudaEventRecord(start);
	//flatFieldCorrect_cpu();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("sequential: %.1f (ms) \n", milli);


	//set up dark and light image, and dimensions for output
	uint32 width, height;
	uint16 bps, spp, photo, sampleFormat;

	TIFF* toCorrect = TIFFOpen("/home/alessandro/Documents/ImageData/100915/lamb/lamb0_00001.tif", "r");
	TIFF *light 		= TIFFOpen("/home/alessandro/Documents/ImageData/100915/MED_light.tif","r");
	TIFF *dark 			= TIFFOpen("/home/alessandro/Documents/ImageData/100915/MED_dark.tif","r");

	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGEWIDTH, &width));
	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGELENGTH, &height));
	assert(TIFFGetField(toCorrect, TIFFTAG_BITSPERSAMPLE, &bps));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLESPERPIXEL, &spp));
	assert(TIFFGetField(toCorrect, TIFFTAG_PHOTOMETRIC, &photo));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLEFORMAT, &sampleFormat));
	assert(TIFFSetField(light, TIFFTAG_SAMPLEFORMAT, 3));
	std::cout << "light sf " << sampleFormat << std::endl;
	assert(TIFFSetField(dark, TIFFTAG_SAMPLEFORMAT, 3));
	std::cout << "dark sf "<< sampleFormat << std::endl;


	std::cout << "width " << width << std::endl;
	std::cout << "height " << height << std::endl;
	std::cout << "bps " << bps << std::endl;
	std::cout << "spp " << spp << std::endl;
	std::cout << "photo " << photo << std::endl;
	std::cout << "sample format " << sampleFormat << std::endl;

	int npixels = width*height;
	int linesize = TIFFScanlineSize(toCorrect);

	TIFFClose(toCorrect);

	cudaEventRecord(start);

	//allocate memory on host
	GreyscaleValue * h_inputData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_lightData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_darkData   = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_correctedData = (GreyscaleValue *) _TIFFmalloc(linesize * width);

	Real h_lightAverage = 0.0;
	for (int row = 0; row < height; row++) {
		assert(TIFFReadScanline(light, &h_lightData[row * linesize], row));
		assert(TIFFReadScanline(dark, &h_darkData[row * linesize], row));
		for (int column = 0; column < width; column++)
		{
			h_lightAverage += (Real) h_lightData[row * linesize+column];
		}
	}
	h_lightAverage /= (npixels);


	//allocate memory on device
	const int dataSize = npixels*sizeof(GreyscaleValue);
	std::cout << "data size " << dataSize << std::endl;
	GreyscaleValue * d_data;
	GreyscaleValue * d_lightData;
	GreyscaleValue * d_darkData;

	GreyscaleValue h_typeMax = std::numeric_limits<GreyscaleValue>::max();
	checkCudaErrors(cudaMalloc( (void**)&d_data, dataSize));
	checkCudaErrors(cudaMalloc( (void**)&d_lightData, dataSize));
	checkCudaErrors(cudaMalloc( (void**)&d_darkData, dataSize));

	checkCudaErrors(cudaMemcpy( d_lightData, h_lightData, dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_darkData,  h_darkData,  dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpyToSymbol(d_lightAverage, &h_lightAverage, sizeof(Real)));
	checkCudaErrors(cudaMemcpyToSymbol(d_typeMax, &h_typeMax, sizeof(GreyscaleValue)));
	std::cout << "data size " << dataSize << std::endl;

	dim3 dimBlock( blocksize, 1);
	dim3 dimGrid( gridsize, 1 );

	int nImages=25994;
	for(int i=1; i<nImages+1; i++)
	{
		char filename[150];
		sprintf (filename, "/home/alessandro/Documents/ImageData/100915/lamb/lamb0_%05d.tif", i);
		TIFF * toCorrectGPU	=	TIFFOpen(filename, "r");
		sprintf (filename, "/home/alessandro/Documents/ImageData/100915/lamb/corrected/lamb0_%05d-correctedGPU.tif", i);
		TIFF *correctedGPU	=	TIFFOpen(filename, "w");

		assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGEWIDTH, width));
		assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGELENGTH, height));
		assert(TIFFSetField(correctedGPU, TIFFTAG_BITSPERSAMPLE, bps));
		assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLESPERPIXEL, spp));
		assert(TIFFSetField(correctedGPU, TIFFTAG_PHOTOMETRIC, photo));
		assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLEFORMAT, sampleFormat));

		for (int row = 0; row < height; row++)
		{
			assert(TIFFReadScanline(toCorrectGPU, &h_inputData[row * linesize], row));
		}

		checkCudaErrors(cudaMemcpy( d_data, h_inputData, dataSize, cudaMemcpyHostToDevice));
		flatFieldCorrect<<<dimGrid, dimBlock>>>(d_data,d_lightData,d_darkData);
		checkCudaErrors(cudaMemcpy(h_correctedData, d_data, dataSize, cudaMemcpyDeviceToHost));

		for (int row = 0; row < height; row++)
		{
			assert(TIFFWriteScanline(correctedGPU, &h_correctedData[row*linesize], row));
		}

		TIFFClose(correctedGPU);
		TIFFClose(toCorrectGPU);
		std::cout << i << std::endl;
	}

	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_lightData));
	checkCudaErrors(cudaFree(d_darkData));


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("\n gpu: %.1f (ms) \n", milli);

	cudaDeviceReset();

	_TIFFfree(h_inputData);
	_TIFFfree(h_correctedData);
	_TIFFfree(h_lightData);
	_TIFFfree(h_darkData);

	TIFFClose(light);
	TIFFClose(dark);

    return 0;

}
