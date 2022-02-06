
#include "MaxPool.cuh"

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
using namespace std;

__global__ void MaxPoolKernel(float* d_Input, float* d_Output, int _inputHeight, int _inputWidth, int _outputHeight, int _outputWidth)
{
	//These define indecies for output matrix;
	int outputRowIndex = blockIdx.x;
	int outputColumnIndex = threadIdx.x;

	//Starts from "top left" of current block of pixels being processed
	int inputRowIndex = blockIdx.x * 2;
	int inputColumnIndex = threadIdx.x * 2;




	int outputArrayIndex = outputRowIndex * _outputWidth + outputColumnIndex;

	int inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;

	float currentMax = d_Input[inputArrayIndex];
	float currentPixel;
	for (int row = 0; row < 2; row++)
	{
		inputColumnIndex = threadIdx.x * 2;
		inputRowIndex += row;


		for (int col = 0; col < 2; col++)
		{
			inputColumnIndex += col;

			inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;

			currentPixel = d_Input[inputArrayIndex];

			//NOTE the case currentPixel >= currentMax
			if (currentPixel > currentMax)
			{
				currentMax = currentPixel;
			}
		}

	}

	d_Output[outputArrayIndex] = currentMax;

};


__global__ void BackpropMaxPoolKernel(float* d_BackpropInput, float* d_FwdInput, float* d_Output, int _fwdInputHeight, int _fwdInputWidth, int _backInputHeight, int _backInputWidth, int _backOutputHeight, int _backOutputWidth)
{
	//These define indecies for output matrix;
	int backInputRowIndex = blockIdx.x; ///OK
	int backInputColumnIndex = threadIdx.x; ///OK

	//Starts from "top left" of current block of pixels being processed
	int fwdInputRowIndex = blockIdx.x * 2; ///OK
	int fwdInputColumnIndex = threadIdx.x * 2; ///OK


	int backInputIndex = backInputRowIndex * _backInputWidth + backInputColumnIndex; ///OK

	int fwdInputArrayIndex = fwdInputRowIndex * _fwdInputWidth + fwdInputColumnIndex; ///OK

	float currentMax = d_FwdInput[fwdInputArrayIndex]; ///OK
	float currentPixel; ///OK
	int maxElementIndex = fwdInputArrayIndex; ///OK

	for (int row = 0; row < 2; row++)
	{
		fwdInputColumnIndex = threadIdx.x * 2; ///OK
		fwdInputRowIndex += row; ///OK


		for (int col = 0; col < 2; col++)
		{
			fwdInputColumnIndex += col; ///OK

			fwdInputArrayIndex = fwdInputRowIndex * _fwdInputWidth + fwdInputColumnIndex; ///OK

			currentPixel = d_FwdInput[fwdInputArrayIndex]; ///OK

			//NOTE the case currentPixel >= currentMax
			if (currentPixel > currentMax) ///OK
			{
				maxElementIndex = fwdInputArrayIndex;
				currentMax = currentPixel; ///OK
			}
		}

	}
	
	d_Output[maxElementIndex] = d_BackpropInput[backInputIndex];
};

MaxPool::MaxPool()
{

}

void MaxPool::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{

	m_InputMatrixHeight = fwdPassHeight;
	m_InputMatrixWidth = fwdPassWidth;

	m_OutputMatrixHeight = m_InputMatrixHeight / 2;
	m_OutputMatrixWidth = m_InputMatrixWidth / 2;

	int arrayLength = fwdPassHeight * fwdPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];

	memcpy(m_InputMatrix, forwardPassInput, inputSize);


	if (m_InputMatrixWidth % 2 != 0 || m_InputMatrixHeight % 2 != 0)
	{
		cout << "Current version of MaxPool does not support odd matrix sizes of type " << m_InputMatrixHeight << "x" << m_InputMatrixWidth << endl;
		return;
	}
	size_t totalInputPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	size_t totalOutPutPixelCount = m_OutputMatrixHeight * m_OutputMatrixWidth;
	int inputByteCount = totalInputPixelCount * sizeof(float);
	int outputByteCount = totalOutPutPixelCount * sizeof(float);

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemset((void*)d_Output, -69, outputByteCount);

	cudaMemcpy(d_Input, m_InputMatrix, inputByteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_InputMatrixHeight / 2, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth / 2, 1, 1);

	MaxPoolKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixHeight, m_InputMatrixWidth, m_OutputMatrixHeight, m_OutputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void MaxPool::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{
	m_BackpropInputMatrixHeight = backPassHeight; ///OK
	m_BackpropInputMatrixWidth = backPassWidth; ///OK

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight * 2; ///OK
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth * 2; ///OK

	int backpropInputArraySize = backPassHeight * backPassWidth; ///OK
	size_t backpropInputArrayByteCount = backpropInputArraySize * sizeof(float); ///OK

	m_BackPropInputMatrix = new float[backpropInputArraySize]; ///OK
	m_BackpropagationOutput= new float[m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth]; ///OK

	memcpy(m_BackPropInputMatrix, backpropInput, backpropInputArrayByteCount); ///OK

	size_t fwdInputPixelCount = m_InputMatrixHeight * m_InputMatrixWidth; ///OK
	size_t backInputPixelCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth; ///OK
	size_t totalOutPutPixelCount = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth; ///OK

	int fwdInputByteCount = fwdInputPixelCount * sizeof(float); ///OK
	int backInputByteCount = backInputPixelCount * sizeof(float); ///OK
	int outputByteCount = totalOutPutPixelCount * sizeof(float); ///OK

	//Define pointers for deviceMemory locations
	float* d_FwdInput; ///OK
	float* d_BackpropInput; ///OK
	float* d_Output; ///OK

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, fwdInputByteCount); ///OK
	cudaMalloc((void**)&d_BackpropInput, backInputByteCount); ///OK
	cudaMalloc((void**)&d_Output, outputByteCount); ///OK

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemset((void*)d_Output, 0, outputByteCount); ///OK

	cudaMemcpy(d_FwdInput, m_InputMatrix, fwdInputByteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_BackpropInput, m_BackPropInputMatrix, backInputByteCount, cudaMemcpyHostToDevice); ///OK

	//Define block size and threads per block.
	dim3 blockGrid(m_BackpropInputMatrixHeight, 1, 1); ///OK
	dim3 threadGrid(m_BackpropInputMatrixWidth, 1, 1); ///OK

	BackpropMaxPoolKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixHeight, m_InputMatrixWidth, m_BackpropInputMatrixHeight, m_BackpropInputMatrixWidth, m_BackpropOutputMatrixHeight, m_BackpropOutputMatrixWidth);
	cudaDeviceSynchronize(); ///OK

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost); ///OK
}

void MaxPool::UpdateModule()
{

}





