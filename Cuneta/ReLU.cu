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
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;
	float fwdInputPixel = d_FwdInput[arrayIndex];
	float backpropInputPixel = d_BackpropInput[arrayIndex];

	float ReLUResult = 0;
	if (fwdInputPixel > 0)
	{
		ReLUResult = 1 * backpropInputPixel;
	}

	d_BackpropOutput[arrayIndex] = ReLUResult;
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

	int arrayLength = backPassHeight * backPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_BackpropInputMatrixHeight = backPassHeight;
	m_BackpropInputMatrixWidth = backPassWidth;

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight;
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth;

	m_BackPropInputMatrix = new float[arrayLength];
	m_BackpropagationOutput = new float[arrayLength];

	memcpy(m_BackpropagationOutput, backpropInput, inputSize);

	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);
	std::cout << "Pixel count " << totalPixelCount;

	//Define pointers for deviceMemory locations
	float* d_BackpropInput;
	float* d_FwdInput;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_BackpropInput, byteCount);
	cudaMalloc((void**)&d_FwdInput, byteCount);
	cudaMalloc((void**)&d_Output, byteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_BackpropInput, m_BackPropInputMatrix, byteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_FwdInput, m_InputMatrix, byteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_BackpropInputMatrixHeight, 1, 1);
	dim3 threadGrid(m_BackpropInputMatrixWidth, 1, 1);

	BackpropReLUKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, byteCount, cudaMemcpyDeviceToHost);
}

void ReLU::UpdateModule()
{

}





