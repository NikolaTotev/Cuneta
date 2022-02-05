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

__global__ void ConvolutionKernel(float* d_Input, float* d_Filter, float* d_Output, int _outputWidth, int _convolutionInputWidth, int _columnShiftsPerBlock, int filterHeight, int filterWidth)
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
	for (int row = 0; row < filterHeight; row++)
	{
		inputColumnIndex = threadIdx.x;

		for (int col = 0; col < filterWidth; col++)
		{
			inputArrayIndex = inputRowIndex * _convolutionInputWidth + inputColumnIndex;

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
	int rowWriteIndex = (blockIdx.x + 1) * 2 + threadIdx.x;
	int columnWriteIndex = 2;

	int inputRowReadIndex = (blockIdx.x * 2) + threadIdx.x;
	int inputColumnReadIndex = 0;

	int arrayPosition = rowWriteIndex * _paddedInputWidth + columnWriteIndex;
	int inputArrayPosition = inputRowReadIndex * _unpaddedInputWidth + inputColumnReadIndex;

	int var = 0;
	for (int i = 0; i < _unpaddedInputWidth; i++)
	{
		d_Output[arrayPosition] = d_UnpaddedInput[inputArrayPosition];
		arrayPosition++;
		inputArrayPosition++;
	}
}


Convolution::Convolution(int _filterSize, int _paddingSize)
{
	filterSize = _filterSize;
	paddingSize = _paddingSize;
	InitializeFilter();
	//TODO FIX INITIALIZATION ===========================================================================
	//TODO FIX INITIALIZATION ===========================================================================
	//TODO FIX INITIALIZATION ===========================================================================
	//TODO FIX INITIALIZATION ===========================================================================
	//TODO FIX INITIALIZATION ===========================================================================
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

	ConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_OutputMatrixWidth, m_InputMatrixWidth, columnShifts, filterSize, filterSize);
	cudaDeviceSynchronize();

	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

}


void Convolution::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{
	m_BackpropInputMatrixHeight = backPassHeight;
	m_BackpropInputMatrixWidth = backPassWidth;

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight + 2;
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth + 2;

	int arrayLength = backPassHeight * backPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize);

	//Main backprop

	int rowShifts = m_BackpropOutputMatrixHeight;
	int columnShifts = m_BackpropOutputMatrixWidth;

	int elementsInInput = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth;
	int elementsInOutput = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth;
	std::cout << "Number of row shifts " << rowShifts << std::endl;
	std::cout << "Number of column shifts " << columnShifts << std::endl;

	std::cout << "Input elements " << elementsInInput << std::endl;
	std::cout << "Output elements" << elementsInOutput << std::endl;

	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t inputElementCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth;
	size_t filterMatrixElementCount = filterSize * filterSize;
	size_t outputElementCount = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth;

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
	cudaMemcpy(d_Input, m_BackPropInputMatrix, inputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Filter, flippedFilter, filterByteCount, cudaMemcpyHostToDevice);

	ConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_OutputMatrixWidth, m_InputMatrixWidth, columnShifts, filterSize, filterSize);
	cudaDeviceSynchronize();

	cudaFree(d_Input);
	cudaFree(d_Filter);
	cudaFree(d_Output);

	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

	//Filter backprop
	FilterBackprop(backpropInput, backPassHeight, backPassWidth);
}

void Convolution::FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth)
{
	size_t fwdInputElementCount = m_InputMatrixHeight * m_InputMatrixWidth;
	size_t filterEqivElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth;
	size_t  filterOutputElementCount = filterSize * filterSize;

	int fwdInputByteCount = fwdInputElementCount * sizeof(float);
	int filterEqivByteCount = filterEqivElementCount * sizeof(float);
	int filterOutputByteCount = filterOutputElementCount * sizeof(float);

	float* d_FwdInput;
	float* d_FilterEquiv;
	float* d_FilterOutput;

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, fwdInputByteCount);
	cudaMalloc((void**)&d_FilterEquiv, filterEqivByteCount);
	cudaMalloc((void**)&d_FilterOutput, filterOutputByteCount);

	//Copy filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_FwdInput, m_InputMatrix, fwdInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterEquiv, m_OutputMatrix, filterEqivByteCount, cudaMemcpyHostToDevice);

	int rowShifts = filterSize;
	int columnShifts = filterSize;

	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	ConvolutionKernel << <blockGrid, threads >> > (d_FwdInput, d_FilterEquiv, d_FilterOutput, filterSize, m_InputMatrixWidth, columnShifts, m_OutputMatrixHeight, m_OutputMatrixWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(filterBackpropResult, d_FilterOutput, filterOutputByteCount, cudaMemcpyDeviceToHost);
}


void Convolution::PadBackpropInput()
{
	paddedInputHeight = m_BackpropInputMatrixHeight + 2 * paddingSize;
	paddedInputWidth = m_BackpropInputMatrixWidth + 2 * paddingSize;
	int elementsInPaddedInput = paddedInputHeight * paddedInputWidth;

	paddedBackpropInput = new float[elementsInPaddedInput];

	memset(paddedBackpropInput, 0, elementsInPaddedInput * sizeof(float));

	float* d_Output;
	float* d_UnpaddedInput;

	size_t outputByteCount = elementsInPaddedInput * sizeof(float);
	size_t unpaddedInputByteCount = (m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth) * sizeof(float);

	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMalloc((void**)&d_UnpaddedInput, unpaddedInputByteCount);

	cudaMemcpy(d_UnpaddedInput, m_BackPropInputMatrix, unpaddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Output, paddedBackpropInput, unpaddedInputByteCount, cudaMemcpyHostToDevice);

	int numberOfBlocks = m_BackpropInputMatrixHeight / 2;
	cout << "Launching blocks: " << numberOfBlocks << endl;
	dim3 blockGrid(numberOfBlocks, 1, 1);
	dim3 threads(2, 1, 1);

	PaddingKernel << <blockGrid, threads >> > (d_UnpaddedInput, d_Output, paddedInputWidth, m_BackpropInputMatrixWidth, m_BackpropInputMatrixHeight);
	cudaDeviceSynchronize();

	cudaMemcpy(paddedBackpropInput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
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
		filter[i] = i + 1;//distribution(gen);  //TODO FIX INITIALIZATION
	}
}

void Convolution::FlipFilter()
{
	int filterArraySize = filterSize * filterSize;
	flippedFilter = new float[filterArraySize];

	int k = 0;

	//Loop from back and assign value to new array
	for (int i = filterArraySize - 1; i >= 0; ) {
		flippedFilter[k++] = filter[i--];
	}
}


void Convolution::UpdateModule()
{

}


