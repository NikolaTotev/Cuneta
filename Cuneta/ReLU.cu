#include "ReLU.cuh"

#include <vector_types.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#ifndef  __CUDART_API_PER_THREAD_DEFAULT_STREAM
#define  __CUDART_API_PER_THREAD_DEFAULT_STREAM
#endif
#include <cuda_runtime_api.h>


#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__

#endif

#include <algorithm>
#include <iostream>

#include "device_launch_parameters.h"



//Input and output will be in global memory. d_ shows in which memory the variables are stored.
__global__ void ReLUKernel(float* d_Input, float* d_Output, int matrixHeight, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int smallerSide = fminf(matrixHeight, matrixWidth);
	int arrayIndex = rowIndex* smallerSide + columnIndex;
	float pixel = d_Input[arrayIndex];

	float ReLUResult = fmaxf(0,pixel);
	
	d_Output[arrayIndex] = ReLUResult;
}

ReLU::ReLU(float* _inputMatrix, float* _outputMatrix, int _inHeight, int _outHeight, int _inWidth, int _outWidth)
{
	m_InputMatrix = _inputMatrix;
	m_OutputMatrix = _outputMatrix;

	m_InputMatrixHeight = _inHeight;
	m_InputMatrixWidth = _inWidth;

	m_OutputMatrixHeight = _outHeight;
	m_OutputMatrixWidth = _outWidth;
}


void ReLU::ForwardPass()
{
	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);
	std::cout << "Pixel count " << totalPixelCount;

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_Input, byteCount);
	cudaMalloc((void**)&d_Output, byteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_InputMatrix, byteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_InputMatrixHeight,1,1);
	dim3 threadGrid(m_InputMatrixWidth,1,1);

	ReLUKernel <<<blockGrid, threadGrid >>> (d_Input, d_Output, m_InputMatrixHeight, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_OutputMatrix, d_Output, byteCount, cudaMemcpyDeviceToHost);

}

void ReLU::BackwardPass()
{
	
}

void ReLU::UpdateModule()
{
	
}





