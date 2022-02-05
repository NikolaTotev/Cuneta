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
#include "TransposeConvolution.cuh"
#include <random>
#include <cmath>
using namespace std;

__global__ void TransposeConvolutionKernel(float* d_Input, float* d_Filter, float* d_Output, int _outputWidth, int _convolutionInputWidth)
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
		inputColumnIndex = threadIdx.x;


		for (int col = 0; col < 3; col++)
		{

			inputArrayIndex = inputRowIndex * _convolutionInputWidth + inputColumnIndex;
			/*if(blockIdx.x==5 && threadIdx.x == 5)
			{
				d_Output[temp] =
					inputArrayIndex;
				temp++;
			}
			*/
			result += d_Input[inputArrayIndex] * d_Filter[filterIndex];
			filterIndex++;
			inputColumnIndex += 1;


		}
		inputRowIndex += 1;


	}

	d_Output[outputArrayIndex] = result;
};

__global__ void PaddingKernel(float* d_UnpaddedInput, float* d_Output, int _paddedInputWidth, int _unpaddedInputWidth, int _unpaddedInputHeight)
{
	int rowWriteIndex = (blockIdx.x + 1) * 2+threadIdx.x;
	int columnWriteIndex = 2;

	int inputRowReadIndex = (blockIdx.x * 2 )+ threadIdx.x;
	int inputColumnReadIndex = 0;

	int arrayPosition = rowWriteIndex * _paddedInputWidth + columnWriteIndex;
	int inputArrayPosition = inputRowReadIndex * _unpaddedInputWidth + inputColumnReadIndex;


	//if(blockIdx.x == 0 && threadIdx.x == 1)
	//{
	//	d_Output[0] = arrayPosition;
	//	d_Output[1] = rowWriteIndex;
	//	d_Output[2] = columnWriteIndex;
	//	d_Output[3] = _paddedInputWidth;
	//	d_Output[4] = threadIdx.x;
	//	d_Output[5] = inputArrayPosition;
	//}
	//
	int var = 0;
	for (int i = 0; i < _unpaddedInputWidth; i++)
	{
		d_Output[arrayPosition] = d_UnpaddedInput[inputArrayPosition];
		arrayPosition++;
		inputArrayPosition++;
		
	}

}

TransposeConvolution::TransposeConvolution(int _filterSize, int _paddingSize)
{
	filterSize = _filterSize;
	paddingSize = _paddingSize;

	InitializeFilter();
	PadInput();
}


void TransposeConvolution::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{

	m_InputMatrixHeight = fwdPassHeight;
	m_InputMatrixWidth = fwdPassWidth;

	m_OutputMatrixHeight = m_InputMatrixHeight + 2;
	m_OutputMatrixWidth = m_InputMatrixWidth + 2;

	int arrayLength = fwdPassHeight * fwdPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];

	memcpy(m_InputMatrix, forwardPassInput, inputSize);



	int rowShifts = m_OutputMatrixHeight;
	int columnShifts = m_OutputMatrixWidth;

	int elementsInPaddedInput = paddedInputHeight * paddedInputWidth;
	int elementsInOutput = m_OutputMatrixHeight * m_OutputMatrixWidth;
	std::cout << "Number of row shifts " << rowShifts << std::endl;
	std::cout << "Number of column shifts " << columnShifts << std::endl;

	std::cout << "Input elements " << elementsInPaddedInput << std::endl;
	std::cout << "Output elements" << elementsInOutput << std::endl;

	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t paddedInputElementCount = paddedInputHeight * paddedInputWidth;
	size_t filterMatrixElementCount = filterSize * filterSize;
	size_t outputElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth;

	int paddedInputByteCount = paddedInputElementCount * sizeof(float);
	int filterByteCount = filterMatrixElementCount * sizeof(float);
	int outputByteCount = outputElementCount * sizeof(float);
	std::cout << "Input element count " << paddedInputElementCount << std::endl;
	std::cout << "Filter element count " << filterMatrixElementCount << std::endl;
	std::cout << "Output element count " << outputByteCount << std::endl;

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Filter;
	float* d_Output;


	//Allocate memory
	cudaMalloc((void**)&d_Input, paddedInputByteCount);
	cudaMalloc((void**)&d_Filter, filterByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);


	//Copy filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, paddedInput, paddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Filter, filter, filterByteCount, cudaMemcpyHostToDevice);

	TransposeConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_OutputMatrixWidth, paddedInputWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void TransposeConvolution::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{

}

void TransposeConvolution::UpdateModule()
{

}

void TransposeConvolution::PadInput()
{
	paddedInputHeight = m_InputMatrixWidth + 2 * paddingSize;
	paddedInputWidth = m_InputMatrixHeight + 2 * paddingSize;
	int elementsInPaddedInput = paddedInputHeight * paddedInputWidth;

	paddedInput = new float[elementsInPaddedInput];

	memset(paddedInput, 0, elementsInPaddedInput * sizeof(float));

	float* d_Output;
	float* d_UnpaddedInput;

	size_t outputByteCount = elementsInPaddedInput * sizeof(float);
	size_t unpaddedInputByteCount = (m_InputMatrixHeight * m_InputMatrixWidth) * sizeof(float);

	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMalloc((void**)&d_UnpaddedInput, unpaddedInputByteCount);

	cudaMemcpy(d_UnpaddedInput, m_InputMatrix, unpaddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Output, paddedInput, unpaddedInputByteCount, cudaMemcpyHostToDevice);

	int numberOfBlocks = m_InputMatrixHeight / 2;
	cout << "Launching blocks: " << numberOfBlocks << endl;
	dim3 blockGrid(numberOfBlocks, 1, 1);
	dim3 threads(2, 1, 1);

	PaddingKernel << <blockGrid, threads >> > (d_UnpaddedInput, d_Output, paddedInputWidth, m_InputMatrixWidth, m_InputMatrixHeight);
	cudaDeviceSynchronize();

	cudaMemcpy(paddedInput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void TransposeConvolution::InitializeFilter()
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