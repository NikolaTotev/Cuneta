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

__global__ void TransposeConvolutionKernel(float* d_Input, float* d_Filter, float* d_Output, int _outputWidth, int _convolutionInputWidth, int filterHeight, int filterWidth)
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

TransposeConvolution::TransposeConvolution(int _filterSize, int _paddingSize)
{
	m_FilterSize = _filterSize;
	m_PaddingSize = _paddingSize;

	InitializeFilter();
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

	PadInput();

	int rowShifts = m_OutputMatrixHeight;
	int columnShifts = m_OutputMatrixWidth;

	int elementsInPaddedInput = m_PaddedInputHeight * m_PaddedInputWidth;
	int elementsInOutput = m_OutputMatrixHeight * m_OutputMatrixWidth;


	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t paddedInputElementCount = m_PaddedInputHeight * m_PaddedInputWidth;
	size_t filterMatrixElementCount = m_FilterSize * m_FilterSize;
	size_t outputElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth;

	int paddedInputByteCount = paddedInputElementCount * sizeof(float);
	int filterByteCount = filterMatrixElementCount * sizeof(float);
	int outputByteCount = outputElementCount * sizeof(float);

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Filter;
	float* d_Output;


	//Allocate memory
	cudaMalloc((void**)&d_Input, paddedInputByteCount);
	cudaMalloc((void**)&d_Filter, filterByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);


	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_PaddedInput, paddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Filter, m_Filter, filterByteCount, cudaMemcpyHostToDevice);

	TransposeConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_OutputMatrixWidth, m_PaddedInputWidth, m_FilterSize, m_FilterSize);
	cudaDeviceSynchronize();

	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void TransposeConvolution::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{
	m_BackpropInputMatrixHeight = backPassHeight; ///OK
	m_BackpropInputMatrixWidth = backPassWidth; ///OK

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight - 2; ///OK
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth - 2; ///OK

	int arrayLength = backPassHeight * backPassWidth; ///OK
	size_t inputSize = arrayLength * sizeof(float); ///OK

	m_BackPropInputMatrix = new float[arrayLength]; ///OK
	m_BackpropagationOutput = new float[m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth]; ///OK

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize); ///OK

	//Main backprop

	int rowShifts = m_BackpropOutputMatrixHeight; ///OK
	int columnShifts = m_BackpropOutputMatrixWidth; ///OK
	
	dim3 blockGrid(rowShifts, 1, 1); ///OK
	dim3 threads(columnShifts, 1, 1); ///OK

	size_t inputElementCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth; ///OK
	size_t filterMatrixElementCount = m_FilterSize * m_FilterSize; ///OK
	size_t outputElementCount = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth; ///OK

	int inputByteCount = inputElementCount * sizeof(float); ///OK
	int filterByteCount = filterMatrixElementCount * sizeof(float); ///OK
	int outputByteCount = outputElementCount * sizeof(float); ///OK

	FlipFilter();  ///OK

	//Define pointers for deviceMemory locations
	float* d_Input; ///OK
	float* d_Filter; ///OK
	float* d_Output; ///OK


	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount); ///OK
	cudaMalloc((void**)&d_Filter, filterByteCount); ///OK
	cudaMalloc((void**)&d_Output, outputByteCount); ///OK


	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_BackPropInputMatrix, inputByteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_Filter, m_FlippedFilter, filterByteCount, cudaMemcpyHostToDevice); ///OK

	TransposeConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_BackpropOutputMatrixWidth, m_BackpropInputMatrixWidth, m_FilterSize, m_FilterSize); ///OK
	cudaDeviceSynchronize(); ///OK

	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost); ///OK

	//Filter backprop
	FilterBackprop(backpropInput, backPassHeight, backPassWidth); ///OK

	cudaFree(d_Input);
	cudaFree(d_Filter);
	cudaFree(d_Output);
}

void TransposeConvolution::FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth)
{
	size_t fwdInputElementCount = m_PaddedInputHeight * m_PaddedInputWidth;  ///OK
	size_t filterEqivElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth; ///OK
	size_t  filterOutputElementCount = m_FilterSize * m_FilterSize; ///OK

	int fwdInputByteCount = fwdInputElementCount * sizeof(float); ///OK
	int filterEqivByteCount = filterEqivElementCount * sizeof(float); ///OK
	int filterOutputByteCount = filterOutputElementCount * sizeof(float); ///OK

	m_FilterBackpropResult = new float[filterOutputElementCount];

	float* d_FwdInput; ///OK
	float* d_FilterEquiv; ///OK
	float* d_FilterOutput; ///OK

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, fwdInputByteCount); ///OK
	cudaMalloc((void**)&d_FilterEquiv, filterEqivByteCount); ///OK
	cudaMalloc((void**)&d_FilterOutput, filterOutputByteCount); ///OK

	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_FwdInput, m_PaddedInput, fwdInputByteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_FilterEquiv, m_BackPropInputMatrix, filterEqivByteCount, cudaMemcpyHostToDevice); ///OK

	int rowShifts = m_FilterSize; ///OK
	int columnShifts = m_FilterSize; ///OK

	dim3 blockGrid(rowShifts, 1, 1); ///OK
	dim3 threads(columnShifts, 1, 1); ///OK

	TransposeConvolutionKernel << <blockGrid, threads >> > (d_FwdInput, d_FilterEquiv, d_FilterOutput, m_FilterSize, m_PaddedInputWidth, m_OutputMatrixHeight, m_OutputMatrixWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(m_FilterBackpropResult, d_FilterOutput, filterOutputByteCount, cudaMemcpyDeviceToHost); ///OK
}


void TransposeConvolution::UpdateModule()
{
	for (int rowIndex = 0; rowIndex < m_FilterSize; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < m_FilterSize; ++columnIndex)
		{
			int index = rowIndex * m_FilterSize + columnIndex;

			float filterBackpropValue = m_FilterBackpropResult[index];
			float oldV = m_AdamOptimizer_VMatrix[index];
			float oldS = m_AdamOptimizer_SMatrix[index];

			float newV = m_HyperParam_Beta1 * oldV + (1 - m_HyperParam_Beta1) * filterBackpropValue;
			float newS = m_HyperParam_Beta2 * oldS + (1 - m_HyperParam_Beta2) * filterBackpropValue;

			float newVCorrected = newV / (1 - pow(m_HyperParam_Beta1, m_HyperParam_T));
			float newSCorrected = newS / (1 - pow(m_HyperParam_Beta2, m_HyperParam_T));

			m_AdamOptimizer_VMatrix[index] = newV;
			m_AdamOptimizer_SMatrix[index] = newS;

			m_AdamOptimizer_Corrected_VMatrix[index] = newVCorrected;
			m_AdamOptimizer_Corrected_SMatrix[index] = newSCorrected;

			float oldFilterValue = m_Filter[index];
			float newF = oldFilterValue - m_HyperParam_alpha * (newVCorrected / sqrt(newSCorrected + m_HyperParam_Epsilon));

			m_Filter[index] = newF;
		}
	}
}

void TransposeConvolution::PadInput()
{
	m_PaddedInputHeight = m_InputMatrixHeight + 2 * m_PaddingSize;
	m_PaddedInputWidth = m_InputMatrixWidth + 2 * m_PaddingSize;
	int elementsInPaddedInput = m_PaddedInputHeight * m_PaddedInputWidth;

	m_PaddedInput = new float[elementsInPaddedInput];

	memset(m_PaddedInput, 0, elementsInPaddedInput * sizeof(float));

	float* d_Output;
	float* d_UnpaddedInput;

	size_t outputByteCount = elementsInPaddedInput * sizeof(float);
	size_t unpaddedInputByteCount = (m_InputMatrixHeight * m_InputMatrixWidth) * sizeof(float);

	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMalloc((void**)&d_UnpaddedInput, unpaddedInputByteCount);

	cudaMemcpy(d_UnpaddedInput, m_InputMatrix, unpaddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Output, m_PaddedInput, unpaddedInputByteCount, cudaMemcpyHostToDevice);

	int numberOfBlocks = m_InputMatrixHeight / 2;

	dim3 blockGrid(numberOfBlocks, 1, 1);
	dim3 threads(2, 1, 1);

	PaddingKernel << <blockGrid, threads >> > (d_UnpaddedInput, d_Output, m_PaddedInputWidth, m_InputMatrixWidth, m_InputMatrixHeight);
	cudaDeviceSynchronize();

	cudaMemcpy(m_PaddedInput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void TransposeConvolution::InitializeFilter()
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> distribution{ 1.42,2 };
	m_Filter = new float[m_FilterSize * m_FilterSize];

	for (int i = 0; i < m_FilterSize * m_FilterSize; ++i)
	{
		m_Filter[i] = distribution(gen);
	}
}

void TransposeConvolution::FlipFilter()
{
	int filterArraySize = m_FilterSize * m_FilterSize;
	m_FlippedFilter = new float[filterArraySize];

	int k = 0;

	//Loop from back and assign value to new array
	for (int i = filterArraySize - 1; i >= 0; ) {
		m_FlippedFilter[k++] = m_Filter[i--];
	}
}