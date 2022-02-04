
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




	int outputArrayIndex = outputRowIndex *_outputWidth + outputColumnIndex;

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


MaxPool::MaxPool(float* _inputMatrix, int _inputHeight, int _inputWidth)
{
	m_InputMatrix = _inputMatrix;
	m_InputMatrixHeight = _inputHeight;
	m_InputMatrixWidth = _inputWidth;

	m_OutputMatrixHeight = _inputHeight / 2;
	m_OutputMatrixWidth = _inputWidth / 2;

	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];
}

void MaxPool::ForwardPass()
{
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

void MaxPool::BackwardPass()
{

}

void MaxPool::UpdateModule()
{

}





