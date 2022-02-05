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
using namespace std;

__global__ void ConvolutionKernel(float* d_Input, float* d_Filter, float* d_Output, int _toeplitzHeight, int _toeplitzWidth, int _outputHeight, int _outputWidth, int _convolutionInputWidth, int _columnShiftsPerBlock)
{
	//These define indecies for output matrix;
	int outputRowIndex = blockIdx.x;
	int outputColumnIndex = threadIdx.x;

	//Starts from "top left" of current block of pixels being processed
	int inputRowIndex = blockIdx.x;
	int inputColumnIndex = threadIdx.x;

	int outputArrayIndex = outputRowIndex * _outputWidth + outputColumnIndex;

	int inputArrayIndex = 0;

	float result = 0;
	int filterIndex = 0;
	int temp = 0;
	for (int row = 0; row < 3; row++)
	{
		inputColumnIndex = threadIdx.x ;

		for (int col = 0; col < 3; col++)
		{
			inputArrayIndex = inputRowIndex * _convolutionInputWidth + inputColumnIndex;

			/*if (blockIdx.x == 1 && threadIdx.x == 1)
			{
				d_Output[temp] =
					inputArrayIndex;
				temp++;
			}*/

			result += d_Input[inputArrayIndex] * d_Filter[filterIndex];
			filterIndex++;
			inputColumnIndex += 1;
		}
		inputRowIndex += 1;

	}

	d_Output[outputArrayIndex] = result;
};

__global__ void ToeplitzKernel(float* d_Filter, float* d_Output, int _toeplitzHeight, int _toeplitzWidth, int _filterHeight, int _filterWidth, int _convolutionInputWidth, int _columnShiftsPerBlock)
{

	int spacing = _convolutionInputWidth - _filterWidth;
	int largerSide = fmaxf(_toeplitzHeight, _toeplitzWidth);
	int initialBlockOffset = blockIdx.x * largerSide * _columnShiftsPerBlock;
	int rowIndexWithinBlock = initialBlockOffset + (threadIdx.x * largerSide);
	int threadWriteIndex = rowIndexWithinBlock + threadIdx.x + (blockIdx.x * _convolutionInputWidth);

	/*if (blockIdx.x ==0 && threadIdx.x == 0)
	{
		d_Output[0] = blockIdx.x;
		d_Output[1] = threadIdx.x;
		d_Output[2] = initialBlockOffset;
		d_Output[3] = rowIndexWithinBlock;
		d_Output[4] = threadWriteIndex;
	}*/

	int counter = 1;
	int filterReadIndex = 0;
	int skippedPositions = 0;
	bool shouldSkip = false;
	for (int j = 0; j < (_filterHeight * _filterWidth) + (2 * spacing); ++j)
	{
		if (counter == 4 || shouldSkip)
		{
			shouldSkip = true;
			skippedPositions++;
			if (skippedPositions == spacing)
			{
				shouldSkip = false;
				skippedPositions = 0;
			}
			counter = 1;

		}
		else
		{
			d_Output[threadWriteIndex] = d_Filter[filterReadIndex];
			filterReadIndex++;
			counter++;
		}
		threadWriteIndex++;
	}
};

Convolution::Convolution(int _filterSize)
{
	filterSize = _filterSize;
	InitializeFilter();
}	


void Convolution::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{

	m_InputMatrixHeight = fwdPassHeight;
	m_InputMatrixWidth = fwdPassWidth;

	m_OutputMatrixHeight = m_InputMatrixHeight - 2;
	m_OutputMatrixWidth = m_InputMatrixWidth - 2;

	int arrayLength = fwdPassHeight * fwdPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];

	memcpy(m_InputMatrix, forwardPassInput, inputSize);



	int rowShifts = m_OutputMatrixHeight;
	int columnShifts = m_OutputMatrixWidth;

	int elementsInInput = m_InputMatrixHeight * m_InputMatrixWidth;
	int elementsInOutput = m_OutputMatrixHeight * m_OutputMatrixWidth;
	std::cout << "Number of row shifts " << rowShifts << std::endl;
	std::cout << "Number of column shifts " << columnShifts << std::endl;

	std::cout << "Input elements " << elementsInInput << std::endl;
	std::cout << "Output elements" << elementsInOutput << std::endl;

	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t inputElementCount = m_InputMatrixHeight * m_InputMatrixWidth;
	size_t filterMatrixElementCount = filterSize * filterSize;
	size_t outputElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth;

	int inputByteCount = inputElementCount * sizeof(float);
	int filterByteCount = filterMatrixElementCount * sizeof(float);
	int outputByteCount = outputElementCount * sizeof(float);
	std::cout << "Input element count " << inputElementCount << std::endl;
	std::cout << "Filter element count " << filterMatrixElementCount << std::endl;
	std::cout << "Output element count " << outputByteCount << std::endl;

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Filter;
	float* d_Output;


	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount);
	cudaMalloc((void**)&d_Filter, filterByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);


	//Copy filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_InputMatrix, inputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Filter, filter, filterByteCount, cudaMemcpyHostToDevice);

	ConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, elementsInInput, elementsInOutput, m_OutputMatrixHeight, m_OutputMatrixWidth, m_InputMatrixWidth, columnShifts);
	cudaDeviceSynchronize();

	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

}


void Convolution::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{

}


void Convolution::Dialate(float* _input, float* _output)
{

}

void Convolution::InitializeFilter()
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> distribution{ 1.42,2 };
	filter = new float[filterSize * filterSize];

	for (int i = 0; i < filterSize * filterSize; ++i)
	{
		filter[i] = i + 1;//distribution(gen);
	}
}


void Convolution::UpdateModule()
{

}


