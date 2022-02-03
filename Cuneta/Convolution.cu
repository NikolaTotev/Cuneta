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
#include "Convolution.cuh"
#include <random>
#include <cmath>

__global__ void ConvolutionKernel(float* d_Input, float* d_Output, int _inputHeight, int _inputWidth, int _outputHeight, int _outputWidth)
{
	//These define indecies for output matrix;
	int outputRowIndex = blockIdx.x;
	int outputColumnIndex = threadIdx.x;

	//Starts from "top left" of current block of pixels being processed
	int inputRowIndex = blockIdx.x * 2;
	int inputColumnIndex = threadIdx.x * 2;


	int smallerInputSide = fminf(_inputHeight, _inputWidth);
	int smallerOutputSide = fminf(_outputHeight, _outputWidth);

	int outputArrayIndex = outputRowIndex * smallerOutputSide + outputColumnIndex;

	int inputArrayIndex = inputRowIndex * smallerInputSide + inputColumnIndex;
	int initialTopLeftRow = inputArrayIndex;

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

			inputArrayIndex = inputRowIndex * smallerInputSide + inputColumnIndex;

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

__global__ void ToeplitzKernel(float* d_Filter, float* d_Output, int _toeplitzHeight, int _toeplitzWidth, int _filterHeight, int _filterWidth, int _convolutionInputWidth)
{

	int initialBlockOffset = blockIdx.x * _convolutionInputWidth;
	int largerSide = fmaxf(_toeplitzHeight, _toeplitzWidth);
	int currentRowStart = (blockIdx.x + threadIdx.x)*largerSide;
	int writeIndex = currentRowStart + initialBlockOffset + threadIdx.x;

	
	int counter = 1;

	for (int j = 0; j < _filterHeight * _filterWidth; ++j)
	{
		if (counter == 3)
		{
			counter = 1;
		}
		else
		{
			//d_Output[writeIndex] = d_Filter[j];
			counter++;
		}
		writeIndex++;
	}
};

Convolution::Convolution(float* _inputMatrix, int _inputHeight, int _inputWidth, int _filterSize)
{
	m_InputMatrix = _inputMatrix;
	m_InputMatrixHeight = _inputHeight;
	m_InputMatrixWidth = _inputWidth;

	m_OutputMatrixHeight = _inputHeight - 2;
	m_OutputMatrixWidth = _inputWidth - 2;

	filterSize = _filterSize;

	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];
	InitilizeFilter();
	FilterToToeplitzMatrix();
}

void Convolution::ForwardPass()
{

}


void Convolution::BackwardPass()
{

}


void Convolution::Dialate(float* _input, float* _output)
{

}


void Convolution::FilterToToeplitzMatrix()
{
	int columnShifts = m_OutputMatrixWidth;
	int rowShifts = m_OutputMatrixHeight;
	int elementsInInput = m_InputMatrixHeight * m_InputMatrixWidth;
	int elementsInOutput = m_OutputMatrixHeight * m_OutputMatrixWidth;
	toeplitzMatrix = new float[elementsInInput * elementsInOutput];

	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t toeplitzMatrixElementCount = filterSize * filterSize;
	size_t filterElementCount = elementsInInput * elementsInOutput;
	int toeplitzByteCount = toeplitzMatrixElementCount * sizeof(float);
	int filterByteCount = filterElementCount * sizeof(float);
	std::cout << "Toeplitz element count " << toeplitzMatrixElementCount;
	std::cout << "Filter element count " << filterElementCount;

	//Define pointers for deviceMemory locations
	float* d_Output;
	float* d_Filter;

	//Allocate memory
	cudaMalloc((void**)&d_Output, toeplitzByteCount);
	cudaMemset((void*)d_Output, 0, toeplitzByteCount);

	cudaMalloc((void**)&d_Filter, filterSize);

	//Copy filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Filter, filter, filterByteCount, cudaMemcpyHostToDevice);

	ToeplitzKernel << <blockGrid, threads >> > (d_Filter, d_Output, elementsInInput, elementsInOutput, filterSize, filterSize, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(toeplitzMatrix, d_Output, toeplitzByteCount, cudaMemcpyDeviceToHost);
}

void Convolution::InitilizeFilter()
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> distribution{ 1.42,2 };
	filter = new float[filterSize * filterSize];

	for (int i = 0; i < filterSize * filterSize; ++i)
	{
		filter[i] = distribution(gen);
	}
}


void Convolution::UpdateModule()
{

}


