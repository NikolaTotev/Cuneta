
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
	float currentPixel = 555;
	int var = 0;
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
	int backInputRowIndex = blockIdx.x;
	int backInputColumnIndex = threadIdx.x;

	//Starts from "top left" of current block of pixels being processed
	int fwdInputRowIndex = blockIdx.x * 2;
	int fwdInputColumnIndex = threadIdx.x * 2;


	int backInputIndex = backInputRowIndex * _backInputWidth + backInputColumnIndex;

	int fwdInputArrayIndex = fwdInputRowIndex * _fwdInputWidth + fwdInputColumnIndex;

	float currentMax = d_FwdInput[fwdInputArrayIndex];
	float currentPixel;
	int maxElementIndex = 0;
	int var = 0;
	for (int row = 0; row < 2; row++)
	{
		fwdInputColumnIndex = threadIdx.x * 2;
		fwdInputRowIndex += row;


		for (int col = 0; col < 2; col++)
		{
			fwdInputColumnIndex += col;

			fwdInputArrayIndex = fwdInputRowIndex * _fwdInputWidth + fwdInputColumnIndex;

			currentPixel = d_FwdInput[fwdInputArrayIndex];

			//NOTE the case currentPixel >= currentMax
			if (currentPixel > currentMax)
			{
				maxElementIndex = fwdInputArrayIndex;
				currentMax = currentPixel;
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
	cout << "Input Pixel count " << totalInputPixelCount << endl;
	cout << "Output Pixel count " << totalOutPutPixelCount << endl;

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
	m_BackpropInputMatrixHeight = backPassHeight;
	m_BackpropInputMatrixWidth = backPassWidth;

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight * 2;
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth * 2;

	int arrayLength = backPassHeight * backPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize);


	if (m_InputMatrixWidth % 2 != 0 || m_InputMatrixHeight % 2 != 0)
	{
		cout << "Current version of MaxPool does not support odd matrix sizes of type " << m_InputMatrixHeight << "x" << m_InputMatrixWidth << endl;
		return;
	}

	size_t fwdInputPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	size_t backInputPixelCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth;
	size_t totalOutPutPixelCount = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth;

	int fwdInputByteCount = fwdInputPixelCount * sizeof(float);
	int backInputByteCount = backInputPixelCount * sizeof(float);
	int outputByteCount = totalOutPutPixelCount * sizeof(float);

	cout << "Input Pixel count " << backInputPixelCount << endl;
	cout << "Output Pixel count " << totalOutPutPixelCount << endl;

	//Define pointers for deviceMemory locations
	float* d_FwdInput;
	float* d_BackpropInput;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, fwdInputByteCount);
	cudaMalloc((void**)&d_BackpropInput, backInputByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemset((void*)d_Output, 0, outputByteCount);

	cudaMemcpy(d_FwdInput, m_InputMatrix, fwdInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_BackpropInput, m_InputMatrix, backInputByteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_BackpropInputMatrixHeight, 1, 1);
	dim3 threadGrid(m_BackpropInputMatrixWidth, 1, 1);

	BackpropMaxPoolKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixHeight, m_InputMatrixWidth, m_BackpropInputMatrixHeight, m_BackpropInputMatrixWidth, m_BackpropOutputMatrixHeight, m_BackpropOutputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void MaxPool::UpdateModule()
{

}





