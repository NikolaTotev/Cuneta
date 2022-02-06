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
__global__ void ReLUKernel(float* d_Input, float* d_Output, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;
	float pixel = d_Input[arrayIndex];

	float ReLUResult = fmaxf(0, pixel);

	d_Output[arrayIndex] = ReLUResult;
}

__global__ void BackpropReLUKernel(float* d_BackpropInput, float* d_FwdInput, float* d_BackpropOutput, int matrixWidth)
{
	int rowIndex = blockIdx.x; ///OK
	int columnIndex = threadIdx.x;///OK
	int arrayIndex = rowIndex * matrixWidth + columnIndex; ///OK
	float fwdInputPixel = d_FwdInput[arrayIndex];///OK
	float backpropInputPixel = d_BackpropInput[arrayIndex];///OK

	float ReLUResult = 0; ///OK
	if (fwdInputPixel > 0)
	{
		ReLUResult = 1 * backpropInputPixel;
	}

	d_BackpropOutput[arrayIndex] = ReLUResult;///OK
}

ReLU::ReLU()
{

}


void ReLU::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{
	int arrayLength = fwdPassHeight * fwdPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrixHeight = fwdPassHeight;
	m_InputMatrixWidth = fwdPassWidth;

	m_OutputMatrixHeight = m_InputMatrixHeight;
	m_OutputMatrixWidth = m_InputMatrixWidth;

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[arrayLength];

	memcpy(m_InputMatrix, forwardPassInput, inputSize);

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
	dim3 blockGrid(m_InputMatrixHeight, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth, 1, 1);

	ReLUKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_OutputMatrix, d_Output, byteCount, cudaMemcpyDeviceToHost);

}

void ReLU::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{

	int arrayLength = backPassHeight * backPassWidth; ///OK
	size_t inputSize = arrayLength * sizeof(float); ///OK

	m_BackpropInputMatrixHeight = backPassHeight; ///OK
	m_BackpropInputMatrixWidth = backPassWidth; ///OK

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight;///OK
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth;///OK

	m_BackPropInputMatrix = new float[arrayLength];///OK
	m_BackpropagationOutput = new float[arrayLength];///OK

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize); ///OK

	size_t totalPixelCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth; ///OK
	int byteCount = totalPixelCount * sizeof(float); ///OK

	//Define pointers for deviceMemory locations
	float* d_FwdInput; ///OK
	float* d_BackpropInput; ///OK
	float* d_Output; ///OK

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, byteCount); ///OK
	cudaMalloc((void**)&d_BackpropInput, byteCount);///OK
	cudaMalloc((void**)&d_Output, byteCount);///OK

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_BackpropInput, m_BackPropInputMatrix, byteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_FwdInput, m_InputMatrix, byteCount, cudaMemcpyHostToDevice); ///OK

	//Define block size and threads per block.
	dim3 blockGrid(m_BackpropInputMatrixHeight, 1, 1); ///OK
	dim3 threadGrid(m_BackpropInputMatrixWidth, 1, 1);///OK

	BackpropReLUKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixWidth); ///OK
	cudaDeviceSynchronize();///OK

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, byteCount, cudaMemcpyDeviceToHost);///OK
}

void ReLU::UpdateModule()
{

}





