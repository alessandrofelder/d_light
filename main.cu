#include <stdio.h>
#include <tiffio.h>
#include <assert.h>
#include <iostream>

#include <cuda.h>
#include <helper_cuda.h>

typedef unsigned char GreyscaleValue; //unsigned char for 8-bit and unsigned short for 16-bit tiff

#include "flatFieldCorrect_cpu.hh"

const int gridsize = 2048;
const int blocksize = 512;

__constant__ double d_lightAverage;

__global__
void correct(GreyscaleValue* d_image,GreyscaleValue* d_lightData,GreyscaleValue* d_darkData)
{
	int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
	double outputVal (((double) (d_image[localIndex] - d_darkData[localIndex]))/ ((double) (d_lightData[localIndex]-d_darkData[localIndex])));
	outputVal *= d_lightAverage;
	d_image[localIndex] = (GreyscaleValue) outputVal;
}

int main(int argc, const char **argv)
{
	TIFF *toCorrect	=	TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008.tif", "r");
	TIFF *corrected	=	TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008-corrected.tif", "w");

	 // initialise card
	findCudaDevice(argc, argv);

	// initialise CUDA timing

	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//sequential reference version
	cudaEventRecord(start);
	flatFieldCorrect_cpu(toCorrect, corrected);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("\n sequential: %.1f (ms) \n", milli);

	TIFF *light = TIFFOpen("/home/alessandro/Documents/ImageData/070915/light-median-gimp.tif","r");
	TIFF *dark = TIFFOpen("/home/alessandro/Documents/ImageData/070915/dark-median-gimp.tif","r");
	TIFF *correctedGPU = TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008-correctedGPU.tif", "w");
	toCorrect	=	TIFFOpen("/home/alessandro/Documents/ImageData/070915/sloth/sloth1_00008.tif", "r");

	uint32 width, height;
	uint16 bps, spp, photo, sampleFormat;
	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGEWIDTH, &width));
	assert(TIFFGetField(toCorrect, TIFFTAG_IMAGELENGTH, &height));
	assert(TIFFGetField(toCorrect, TIFFTAG_BITSPERSAMPLE, &bps));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLESPERPIXEL, &spp));
	assert(TIFFGetField(toCorrect, TIFFTAG_PHOTOMETRIC, &photo));
	assert(TIFFGetField(toCorrect, TIFFTAG_SAMPLEFORMAT, &sampleFormat));

	assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGEWIDTH, width));
	assert(TIFFSetField(correctedGPU, TIFFTAG_IMAGELENGTH, height));
	assert(TIFFSetField(correctedGPU, TIFFTAG_BITSPERSAMPLE, bps));
	assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLESPERPIXEL, spp));
	assert(TIFFSetField(correctedGPU, TIFFTAG_PHOTOMETRIC, photo));
	assert(TIFFSetField(correctedGPU, TIFFTAG_SAMPLEFORMAT, sampleFormat));


	int npixels = width*height;

	int linesize = TIFFScanlineSize(toCorrect);

	GreyscaleValue * h_inputData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_lightData  = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_darkData   = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_outputData = (GreyscaleValue *) _TIFFmalloc(linesize * width);

	double h_lightAverage = 0.0;

	for (int row = 0; row < height; row++) {
		assert(TIFFReadScanline(light, &h_lightData[row * linesize], row));
		assert(TIFFReadScanline(dark, &h_darkData[row * linesize], row));
		assert(TIFFReadScanline(toCorrect, &h_inputData[row * linesize], row));
		for (int column = 0; column < width; column++) {
			h_lightAverage += (double) h_lightData[row * linesize+column];
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

	checkCudaErrors(cudaMemcpy( d_data, h_inputData, dataSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy( d_lightData, h_lightData, dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_darkData,  h_darkData,  dataSize, cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpyToSymbol(d_lightAverage, &h_lightAverage, sizeof(double)));

	dim3 dimBlock( blocksize, 1);
	dim3 dimGrid( gridsize, 1 );
	correct<<<dimGrid, dimBlock>>>(d_data,d_lightData,d_darkData);

	checkCudaErrors(cudaMemcpy(h_outputData, d_data, dataSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_data));


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	printf("\n gpu: %.1f (ms) \n", milli);

	for(int row=0; row<height; row++)
	{
		assert(TIFFWriteScanline(correctedGPU, &h_outputData[row*linesize], row));
	}

	cudaDeviceReset();

	_TIFFfree(h_inputData);
	_TIFFfree(h_lightData);
	_TIFFfree(h_darkData);
	_TIFFfree(h_outputData);


	TIFFClose(light);
	TIFFClose(dark);
	TIFFClose(toCorrect);
	TIFFClose(correctedGPU);

    return 0;

}