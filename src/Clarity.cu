#include "Clarity.h"
#include <cstdio>

__device__ unsigned int blkCounter = 0;
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
__global__ void cuBrenner(ulong2 *inData, ulong2 *outData, unsigned int n)
{
	__shared__ ulong2 sdata[1024];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	unsigned int indexWithOffset = idx + blockDim.x;
	if (idx >= n) {
		sdata[tid].x = sdata[tid].y= 0;
	}
	else if (indexWithOffset >= n) {
		sdata[tid].x = inData[idx].x;
		sdata[tid].y = inData[idx].y;
	}
	else {
		sdata[tid].x = inData[idx].x + inData[indexWithOffset].x;
		sdata[tid].y = inData[idx].y + inData[indexWithOffset].y;
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s) {
			sdata[tid].x += sdata[tid + s].x;
			sdata[tid].y += sdata[tid + s].y;
		}
		__syncthreads();
	}
	
	if (tid < 32) {
		sdata[tid].x += sdata[tid + 32].x;
		sdata[tid].y += sdata[tid + 32].y;
		sdata[tid].x += sdata[tid + 16].x;
		sdata[tid].y += sdata[tid + 16].y;
		sdata[tid].x += sdata[tid + 8].x;
		sdata[tid].y += sdata[tid + 8].y;
		sdata[tid].x += sdata[tid + 4].x;
		sdata[tid].y += sdata[tid + 4].y;
		sdata[tid].x += sdata[tid + 2].x;
		sdata[tid].y += sdata[tid + 2].y;
		sdata[tid].x += sdata[tid + 1].x;
		sdata[tid].y += sdata[tid + 1].y;
	}
	// write result for this block to global mem
	if (tid == 0) outData[blockIdx.x] = sdata[0];
}

__global__ void cuBrennerInit(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int gridWidth, const int numBlocks)
{
	__shared__ unsigned int blockIndex;
	__shared__ unsigned int blockX, blockY;
	//__shared__ float sdata[1024];
	//__shared__ float sdata2[1024];
	// loop until all blocks completed
	while (1)
	{
		if ((threadIdx.x == 0) && (threadIdx.y == 0))
		{
			blockIndex = atomicAdd(&blkCounter, 1);// get block to process
			blockX = blockIndex % gridWidth;            // note: this is slow, but only called once per block here
			blockY = blockIndex / gridWidth;
		}
		__syncthreads();
		if (blockIndex >= numBlocks)
		{
			break;    // finish
		}
		const int ix = blockDim.x * blockX + threadIdx.x;
		const int iy = blockDim.y * blockY + threadIdx.y;

		if ((ix < width - 2) && (iy < height))
		{
			const int pixel = width * iy + ix;
			const int dstpixel = (width-2) * iy + ix;
			pTmp[dstpixel].x = pGray[pixel];
			int d = pGray[pixel + 2] - pGray[pixel];
			pTmp[dstpixel].y = d*d;
		}
		//__syncthreads();
	}
}

__global__ void cuBrennerInitWH(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int gridWidth, const int numBlocks)
{
	__shared__ unsigned int blockIndex;
	__shared__ unsigned int blockX, blockY;
	// loop until all blocks completed
	while (1)
	{
		if ((threadIdx.x == 0) && (threadIdx.y == 0))
		{
			blockIndex = atomicAdd(&blkCounter, 1);// get block to process
			blockX = blockIndex % gridWidth;            // note: this is slow, but only called once per block here
			blockY = blockIndex / gridWidth;
		}
		__syncthreads();
		if (blockIndex >= numBlocks)
		{
			break;    // finish
		}
		const int ix = blockDim.x * blockX + threadIdx.x;
		const int iy = blockDim.y * blockY + threadIdx.y;

		if ((ix < width - 2) && (iy < height-2))
		{
			const int pixel = width * iy + ix;
			const int dstPixel = (width-2) * iy + ix;
			pTmp[dstPixel].x = pGray[pixel];
			int dx = pGray[pixel + 2] - pGray[pixel];
			int dy = pGray[pixel + width*2] - pGray[pixel];
			pTmp[dstPixel].y = dx*dx + dy*dy;
			//int item = abs(dx) + abs(dy);
			//if (item == 0)
			//{
			//	pTmp[dstPixel].x = 0;
			//}
			//else
			//{
			//	//去除0梯度 防止波动
			//	pTmp[dstPixel].x = 1000;
			//}
			//if (item > 9)
			//{
			//	pTmp[dstPixel].y = dx * dx + dy * dy;
			//}
			//else
			//{
			//	//筛除低梯度，确保少组织也能对焦
			//	pTmp[dstPixel].y = 0;
			//}
		}
	}
}

bool RunCuBrenner(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int numSMs, int version, int brennerType)
{
	// 检查内存分配
    if (pTmp == nullptr) {
        fprintf(stderr, "pTmp allocation failed\n");
        return false;
    }
	int blockdim_x_dynamic = BLOCKDIM_X;
	int blockdim_y_dynamic = BLOCKDIM_Y;
	if (version >= 20) //override original 1.x block dimensions on newer architectures, supporting 1024 threads/block with warpSize=32
	{
		blockdim_x_dynamic = BLOCKDIM_X_SM20;
		blockdim_y_dynamic = BLOCKDIM_Y_SM20;
	}
	if (brennerType < 1 || brennerType>3) brennerType = 1;
	dim3 threads(blockdim_x_dynamic, blockdim_y_dynamic);
	dim3 grid(iDivUp(width, blockdim_x_dynamic), iDivUp(height, blockdim_y_dynamic));
	int numWorkerBlocks = numSMs;
	unsigned int hBlkCounter = 0;     // zero block counter
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpyToSymbol(blkCounter, &hBlkCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	if (brennerType == 3) {
		cuBrennerInitWH <<<numWorkerBlocks, threads>>>(pGray, pTmp, width, height, grid.x, grid.x *grid.y);
	} else {
		cuBrennerInit <<<numWorkerBlocks, threads>>>(pGray, pTmp, width, height, grid.x, grid.x *grid.y);
	}
	checkCudaErrors(cudaDeviceSynchronize());

	int iMatrixSize = brennerType == 3? (width-2)*(height-2): (width - 2)*height;
	int iNum = iMatrixSize;
	for (int i = 1; i < iMatrixSize; i = 2 * i * BLOCK_SIZE)
	{
		int iBlockNum = (iNum + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);
		cuBrenner <<<iBlockNum, BLOCK_SIZE>>>(pTmp, pTmp, iNum);
		iNum = iBlockNum;
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaMemcpyFromSymbol(&hBlkCounter, blkCounter, sizeof(unsigned int)));
	if (hBlkCounter == 0) {
		fprintf(stderr, "RunCuBrenner not run\n");
		return false;
	}
	return true;
}
