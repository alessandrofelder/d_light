void flatFieldCorrect_cpu() {

	TIFF *light 		= TIFFOpen("/home/alessandro/Documents/ImageData/100915/MED_light.tif","r");
	TIFF *dark 			= TIFFOpen("/home/alessandro/Documents/ImageData/100915/MED_dark.tif","r");
	TIFF *output 		= TIFFOpen("/home/alessandro/Documents/ImageData/100915/lamb/corrected/lamb0_00001-correctedCPU.tif", "w");
	TIFF *input			= TIFFOpen("/home/alessandro/Documents/ImageData/100915/lamb/lamb0_00001.tif", "r");


	uint32 width, height;
	uint16  spp, bps, photo, sampleFormat;
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

	int linesize = TIFFScanlineSize(input);

	GreyscaleValue * h_inputData = (GreyscaleValue *) _TIFFmalloc(linesize);
	GreyscaleValue * h_lightData = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_darkData = (GreyscaleValue *) _TIFFmalloc(linesize * width);
	GreyscaleValue * h_outputData = (GreyscaleValue *) _TIFFmalloc(linesize);

	double h_lightAverage = 0.0;

	for (int row = 0; row < height; row++) {
		TIFFReadScanline(light, &h_lightData[row * linesize], row);
		for (int column = 0; column < width; column++) {
			h_lightAverage += (double) h_lightData[row * linesize+column];
		}
	}

	for (int row = 0; row < height; row++) {
		TIFFReadScanline(dark, &h_darkData[row * linesize], row);
	}

	h_lightAverage /= (width * height);

	for (int row = 0; row < height; row++) {
		TIFFReadScanline(input, h_inputData, row);
		for (int column = 0; column < width; column++) {
			double outputVal = (((double) (h_inputData[column]
					- h_darkData[row * linesize + column]))
					/ ((double) (h_lightData[row * linesize + column]
							- h_darkData[row * linesize + column])));
			outputVal *= h_lightAverage;
			h_outputData[column] = (GreyscaleValue) outputVal;
		}
		TIFFWriteScanline(output, h_outputData, row);
	}
	_TIFFfree(h_inputData);
	_TIFFfree(h_lightData);
	_TIFFfree(h_darkData);
	_TIFFfree(h_outputData);

	TIFFClose(light);
	TIFFClose(dark);
	TIFFClose(input);
	TIFFClose(output);
}
