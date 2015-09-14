
__constant__ int d_typeMax;

__global__
void invert(GreyscaleValue* d_image)
{
	int localIndex = blockIdx.x * blockDim.x + threadIdx.x;

	d_image[localIndex] = d_typeMax - d_image[localIndex];
}
