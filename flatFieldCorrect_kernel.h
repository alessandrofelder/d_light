
__constant__ Real d_lightAverage;

__global__
void flatFieldCorrect(GreyscaleValue* d_image,GreyscaleValue* d_lightData,GreyscaleValue* d_darkData)
{
	int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
	Real outputVal (((Real) (d_image[localIndex] - d_darkData[localIndex]))/ ((Real) (d_lightData[localIndex]-d_darkData[localIndex])));
	outputVal *= d_lightAverage;
	d_image[localIndex] = (GreyscaleValue) outputVal;
}
