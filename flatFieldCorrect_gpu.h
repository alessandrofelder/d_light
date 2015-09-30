#include "fieldImages.h"
#include "flatFieldCorrect_kernel.h"

const int blocksize = 256;

template<typename GreyscaleValue, typename Real> void flatFieldCorrect_gpu(
		FieldImages& fi, const char* fileToCorrectPrefix, const int nImages) {

	char file[200];

	for (int image = 1; image < nImages + 1; image++) {

		snprintf(file, sizeof(file), "%s%05d%s", fileToCorrectPrefix, image,
				".tif");
		TIFF *input = TIFFOpen(file, "r");

		snprintf(file, sizeof(file), "%s%05d%s", fileToCorrectPrefix, image,
				"_correctedGPU.tif");
		TIFF *output = TIFFOpen(file, "w");

		uint32 width, height;
		uint16 spp, bps, photo, sampleFormat;
		TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &width);
		TIFFGetField(input, TIFFTAG_IMAGELENGTH, &height);
		TIFFGetField(input, TIFFTAG_BITSPERSAMPLE, &bps);
		TIFFGetField(input, TIFFTAG_SAMPLESPERPIXEL, &spp);
		TIFFGetField(input, TIFFTAG_PHOTOMETRIC, &photo);
		TIFFGetField(input, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

		TIFFSetField(output, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(output, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(output, TIFFTAG_BITSPERSAMPLE, bps);
		TIFFSetField(output, TIFFTAG_SAMPLESPERPIXEL, spp);
		TIFFSetField(output, TIFFTAG_PHOTOMETRIC, photo);
		TIFFSetField(output, TIFFTAG_SAMPLEFORMAT, sampleFormat);

		assert(bps == 8 * sizeof(GreyscaleValue));

		uint16 linesize = TIFFScanlineSize(input);
		const int dataSize = linesize * height * sizeof(GreyscaleValue);

		GreyscaleValue * h_data = (GreyscaleValue *) _TIFFmalloc(
				dataSize);
		GreyscaleValue * h_darkData = (GreyscaleValue *) _TIFFmalloc(
				dataSize);
		GreyscaleValue * h_lightData = (GreyscaleValue *) _TIFFmalloc(
				dataSize);

		Real h_flatAverage = 0.0;

		for (int row = 0; row < height; row++) {
			TIFFReadScanline(fi.dark, &h_darkData[row * linesize], row);
		}

		for (int row = 0; row < height; row++) {
			TIFFReadScanline(fi.light, &h_lightData[row * linesize], row);
			for (int column = 0; column < width; column++) {
				h_flatAverage += (Real) h_lightData[row * linesize + column]
						- (Real) h_darkData[row * linesize + column];
			}
		}

		h_flatAverage /= (width * height);

		for (int row = 0; row < height; row++) {
			TIFFReadScanline(input, &h_data[row * linesize], row);
		}

		GreyscaleValue * d_data;
		GreyscaleValue * d_lightData;
		GreyscaleValue * d_darkData;

		checkCudaErrors(cudaMalloc( (void**)&d_data, dataSize));
		checkCudaErrors(cudaMalloc( (void**)&d_lightData, dataSize));
		checkCudaErrors(cudaMalloc( (void**)&d_darkData, dataSize));

		checkCudaErrors(cudaMemcpy( d_lightData, h_lightData, dataSize, cudaMemcpyHostToDevice ));
		checkCudaErrors(cudaMemcpy( d_darkData,  h_darkData,  dataSize, cudaMemcpyHostToDevice ));
		checkCudaErrors(cudaMemcpyToSymbol(d_flatAverage, &h_flatAverage, sizeof(Real)));

		dim3 dimBlock( blocksize, 1);
		const int gridsize = width*height*sizeof(GreyscaleValue)/blocksize;
		dim3 dimGrid( gridsize , 1 );

		checkCudaErrors(cudaMemcpy( d_data, h_data, dataSize, cudaMemcpyHostToDevice));
		flatFieldCorrect<GreyscaleValue, Real><<<dimGrid, dimBlock>>>(d_data,d_lightData,d_darkData);
		checkCudaErrors(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost));


		checkCudaErrors(cudaFree(d_data));
		checkCudaErrors(cudaFree(d_lightData));
		checkCudaErrors(cudaFree(d_darkData));

		for (int row = 0; row < height; row++) {
			TIFFWriteScanline(output, &h_data[row * linesize], row);
		}

		_TIFFfree(h_data);
		_TIFFfree(h_lightData);
		_TIFFfree(h_darkData);

		TIFFClose(input);
		TIFFClose(output);
	}
}
