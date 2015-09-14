#include <stdio.h>
#include <tiffio.h>
#include <assert.h>
#include <iostream>
#include <limits>
#include <string>

#include <cuda.h>
#include <helper_cuda.h>

typedef unsigned char GreyscaleValue; //unsigned char for 8-bit and unsigned short for 16-bit tiff
typedef double Real;

#include "flatFieldCorrect_cpu.h"
#include "flatFieldCorrect_kernel.h"

#include "invert_kernel.h"

const int gridsize = 2048;
const int blocksize = 512;


int main(int argc, const char **argv)
{

	 // initialise card
	findCudaDevice(argc, argv);

	// initialise CUDA timing

	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	TIFF *light 		= TIFFOpen("/home/alessandro/Documents/ImageData/070915/light-median-gimp.tif","r");
	TIFF *dark 			= TIFFOpen("/home/alessandro/Documents/ImageData/070915/dark-median-gimp.tif","r");
	TIFF *corrected 	= TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008-corrected.tif", "w");
	TIFF *invertedGPU 	= TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008-invertedGPU.tif", "w");
	TIFF *toCorrect		= TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008.tif", "r");

	//sequential reference version
	cudaEventRecord(start);
	flatFieldCorrect_cpu(toCorrect, corrected);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("\n sequential: %.1f (ms) \n", milli);

	uint32 width, height;
	uint16 bps, spp, photo, sampleFormat;

	toCorrect		= TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008.tif", "r");
	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGEWIDTH, &width));
	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGELENGTH, &height));
	assert(TIFFGetField(toCorrect, TIFFTAG_BITSPERSAMPLE, &bps));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLESPERPIXEL, &spp));
	assert(TIFFGetField(toCorrect, TIFFTAG_PHOTOMETRIC, &photo));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLEFORMAT, &sampleFormat));

	assert(TIFFSetField(invertedGPU, TIFFTAG_IMAGEWIDTH, width));
	assert(TIFFSetField(invertedGPU, TIFFTAG_IMAGELENGTH, height));
	assert(TIFFSetField(invertedGPU, TIFFTAG_BITSPERSAMPLE, bps));
	assert(TIFFSetField(invertedGPU, TIFFTAG_SAMPLESPERPIXEL, spp));
	assert(TIFFSetField(invertedGPU, TIFFTAG_PHOTOMETRIC, photo));
	assert(TIFFSetField(invertedGPU, TIFFTAG_SAMPLEFORMAT, sampleFormat));

	int npixels = width*height;

	int linesize = TIFFScanlineSize(toCorrect);

	GreyscaleValue * h_inputData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_lightData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_darkData   = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_correctedData = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_invertedData = (GreyscaleValue *) _TIFFmalloc(linesize * width);

	Real h_lightAverage = 0.0;

	for (int row = 0; row < height; row++) {
		assert(TIFFReadScanline(light, &h_lightData[row * linesize], row));
		assert(TIFFReadScanline(dark, &h_darkData[row * linesize], row));
		for (int column = 0; column < width; column++) {
			h_lightAverage += (Real) h_lightData[row * linesize+column];
		}
	}

	h_lightAverage /= (npixels);

	const int dataSize = npixels*sizeof(GreyscaleValue);
	GreyscaleValue * d_data;
	GreyscaleValue * d_lightData;
	GreyscaleValue * d_darkData;

	cudaEventRecord(start);

	checkCudaErrors(cudaMalloc( (void**)&d_data, dataSize));
	checkCudaErrors(cudaMalloc( (void**)&d_lightData, dataSize));
	checkCudaErrors(cudaMalloc( (void**)&d_darkData, dataSize));

	checkCudaErrors(cudaMemcpy( d_lightData, h_lightData, dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_darkData,  h_darkData,  dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpyToSymbol(d_lightAverage, &h_lightAverage, sizeof(Real)));

	int h_typeMax = std::numeric_limits<GreyscaleValue>::max();
	checkCudaErrors(cudaMemcpyToSymbol(d_typeMax, &h_typeMax, sizeof(int)));

	dim3 dimBlock( blocksize, 1);
	dim3 dimGrid( gridsize, 1 );

	int nImages=1481;
	for(int i=1; i<nImages+1; i++)
	{
		TIFFClose(toCorrect);
		char filename[100];
		sprintf (filename, "/home/alessandro/Documents/ImageData/070915/sloth/sloth1_%05d.tif", i);
		toCorrect	=	TIFFOpen(filename, "r");
		sprintf (filename, "/home/alessandro/Documents/ImageData/070915/sloth/corrected/sloth1_%05d-correctedGPU.tif", i);
		TIFF *correctedGPU	=	TIFFOpen(filename, "w");

		assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGEWIDTH, width));
		assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGELENGTH, height));
		assert(TIFFSetField(correctedGPU, TIFFTAG_BITSPERSAMPLE, bps));
		assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLESPERPIXEL, spp));
		assert(TIFFSetField(correctedGPU, TIFFTAG_PHOTOMETRIC, photo));
		assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLEFORMAT, sampleFormat));

		for (int row = 0; row < height; row++)
		{
			assert(TIFFReadScanline(toCorrect, &h_inputData[row * linesize], row));
		}

		checkCudaErrors(cudaMemcpy( d_data, h_inputData, dataSize, cudaMemcpyHostToDevice));
		flatFieldCorrect<<<dimGrid, dimBlock>>>(d_data,d_lightData,d_darkData);
		checkCudaErrors(cudaMemcpy(h_correctedData, d_data, dataSize, cudaMemcpyDeviceToHost));

		for (int row = 0; row < height; row++)
		{
		assert(TIFFWriteScanline(correctedGPU, &h_correctedData[row*linesize], row));
		}

		TIFFClose(correctedGPU);
	}
	//checkCudaErrors(cudaMemcpy( d_data, h_inputData, dataSize, cudaMemcpyHostToDevice));
	//invert<<<dimGrid, dimBlock>>>(d_data);
	//checkCudaErrors(cudaMemcpy(h_invertedData, d_data, dataSize, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_lightData));
	checkCudaErrors(cudaFree(d_darkData));


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("\n gpu: %.1f (ms) \n", milli);

	for(int row=0; row<height; row++)
	{
		assert(TIFFWriteScanline(invertedGPU, &h_invertedData[row*linesize], row));

	}

	cudaDeviceReset();

	_TIFFfree(h_inputData);
	_TIFFfree(h_lightData);
	_TIFFfree(h_darkData);
	_TIFFfree(h_correctedData);
	_TIFFfree(h_invertedData);



	TIFFClose(light);
	TIFFClose(dark);
	TIFFClose(invertedGPU);

    return 0;

}
