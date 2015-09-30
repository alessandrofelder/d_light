#include "fieldImages.h"


template<typename GreyscaleValue, typename Real> void flatFieldCorrect_cpu(
		FieldImages& fi, const char* fileToCorrectPrefix, const int nImages) {

	char file[200];

	for (int image = 1; image < nImages + 1; image++) {

		snprintf(file, sizeof(file), "%s%05d%s", fileToCorrectPrefix, image,
				".tif");
		TIFF *input = TIFFOpen(file, "r");

		snprintf(file, sizeof(file), "%s%05d%s", fileToCorrectPrefix, image,
				"_correctedCPU.tif");
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

		GreyscaleValue * h_inputData = (GreyscaleValue *) _TIFFmalloc(
				linesize * sizeof(GreyscaleValue));
		GreyscaleValue * h_darkData = (GreyscaleValue *) _TIFFmalloc(
				linesize * height * sizeof(GreyscaleValue));
		GreyscaleValue * h_lightData = (GreyscaleValue *) _TIFFmalloc(
				linesize * height * sizeof(GreyscaleValue));
		GreyscaleValue * h_outputData = (GreyscaleValue *) _TIFFmalloc(
				linesize * sizeof(GreyscaleValue));

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
			TIFFReadScanline(input, h_inputData, row);
			for (int column = 0; column < width; column++) {
				Real outputVal = (((Real) (h_inputData[column]
						- h_darkData[row * linesize + column]))
						/ ((Real) (h_lightData[row * linesize + column]
								- h_darkData[row * linesize + column])));
				outputVal *= h_flatAverage;
				h_outputData[column] = (GreyscaleValue) outputVal;
			}
			TIFFWriteScanline(output, h_outputData, row);
		}

		_TIFFfree(h_inputData);
		_TIFFfree(h_lightData);
		_TIFFfree(h_darkData);
		_TIFFfree(h_outputData);

		TIFFClose(input);
		TIFFClose(output);
	}
}
