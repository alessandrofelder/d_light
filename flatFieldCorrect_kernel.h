#ifndef FLATFIELDCORRECT_KERNEL_H
#define FLATFIELDCORRECT_KERNEL_H

static __constant__ double d_flatAverage;

template<typename GreyscaleValue, typename Real>
static __global__
void flatFieldCorrect(GreyscaleValue* d_image,GreyscaleValue* d_lightData,GreyscaleValue* d_darkData)
{
	int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
	Real outputVal (((Real) (d_image[localIndex] - d_darkData[localIndex]))/ ((Real) (d_lightData[localIndex]-d_darkData[localIndex])));
	outputVal *= (Real) d_flatAverage;
	d_image[localIndex] = (GreyscaleValue) outputVal;
}

#endif
