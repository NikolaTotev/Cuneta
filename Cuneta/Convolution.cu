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

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif

#include <device_functions.h>
using namespace std;

__global__ void ConvolutionKernel(float* d_Input, float* d_Filter, float* d_Output, int _outputWidth, int _convolutionInputWidth, int filterHeight, int filterWidth)
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
}

__global__ void LayerConvolutionKernal(float** _inputs, float** _filters, float** _outputs, float* biases, int _numberOfInputs, int _outputWidth, int _inputWidth, int _filterSize)
{
	int inputSelectionIndex = threadIdx.x;
	int filterSelectionIndex = blockIdx.z* _numberOfInputs + threadIdx.x;
	int outputSelectionIndex = blockIdx.z;

	float* selectedInput = _inputs[inputSelectionIndex];
	float* selectedFilter = _filters[filterSelectionIndex];
	float* selectedOutput = _outputs[outputSelectionIndex];

	int inputStartReadRowIndex = blockIdx.x;
	int inputStartReadColumnIndex = blockIdx.y;

	int outputWriteRowIndex = blockIdx.x;
	int outputWriteColumnIndex = blockIdx.y;

	int inputArrayIndex = 0;

	int outputArrayIndex = outputWriteRowIndex * _outputWidth + outputWriteColumnIndex;

	float result = 0;
	int filterIndex = 0;
	int temp = 0;
	for (int row = 0; row < _filterSize; row++)
	{
		inputStartReadColumnIndex = blockIdx.y;

		for (int col = 0; col < _filterSize; col++)
		{
			inputArrayIndex = inputStartReadRowIndex * _inputWidth + inputStartReadColumnIndex;

			result += selectedInput[inputArrayIndex] * selectedFilter[filterIndex];
			filterIndex++;
			inputStartReadColumnIndex += 1;
		}
		inputStartReadRowIndex += 1;
	}
	
	//selectedOutput[outputArrayIndex] = filterSelectionIndex;
	atomicAdd(&selectedOutput[outputArrayIndex], result);
};


__global__ void ConvolutionPaddingKernel(float* d_UnpaddedInput, float* d_Output, int _paddedInputWidth, int _unpaddedInputWidth, int _unpaddedInputHeight)
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


Convolution::Convolution(int _filterSize, int _paddingSize, int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth)
{
	m_FilterSize = _filterSize;
	m_PaddingSize = _paddingSize;
	m_AdamOptimizer_VMatrix = new float[m_FilterSize * m_FilterSize];
	m_AdamOptimizer_SMatrix = new float[m_FilterSize * m_FilterSize];

	m_AdamOptimizer_Corrected_VMatrix = new float[m_FilterSize * m_FilterSize];
	m_AdamOptimizer_Corrected_SMatrix = new float[m_FilterSize * m_FilterSize];

	memset(m_AdamOptimizer_VMatrix, 0, m_FilterSize * m_FilterSize * sizeof(float));
	memset(m_AdamOptimizer_SMatrix, 0, m_FilterSize * m_FilterSize * sizeof(float));

	L_FORWARD_NumberOf_INPUTS = _numberOfInputs;
	L_FORWARD_NumerOf_OUTPUTS = _numberOfOutputs;

	L_BACKWARD_NumberOf_INPUTS = L_FORWARD_NumerOf_OUTPUTS;
	L_BACKWARD_NumberOf_OUTPUTS = L_FORWARD_NumberOf_INPUTS;

	L_FORWARD_InputLayer_HEIGHT = _inputHeight;
	L_FORWARD_InputLayer_WIDTH = _inputWidth;

	L_FORWARD_OutputLayer_HEIGHT = _inputHeight / 2;
	L_FORWARD_OutputLayer_WIDTH = _inputWidth / 2;

	L_BACKWARD_InputLayer_HEIGHT = L_FORWARD_OutputLayer_HEIGHT;
	L_BACKWARD_InputLayer_WIDTH = L_FORWARD_OutputLayer_WIDTH;

	L_BACKWARD_OutputLayer_HEIGHT = L_FORWARD_InputLayer_HEIGHT;
	L_BACKWARD_OutputLayer_WIDTH = L_FORWARD_InputLayer_WIDTH;

	L_FORWARD_Pass_INPUTS = new float* [L_FORWARD_NumberOf_INPUTS];
	L_FORWARD_Pass_OUTPUTS = new float* [L_FORWARD_NumerOf_OUTPUTS];

	L_BACKWARD_Pass_INPUTS = new float* [L_FORWARD_NumerOf_OUTPUTS];
	L_BACKWARD_Pass_OUTPUTS = new float* [L_FORWARD_NumberOf_INPUTS];

	L_NumberOf_FILTERS = L_FORWARD_NumberOf_INPUTS * L_FORWARD_NumerOf_OUTPUTS;

	L_Filters = new float* [L_NumberOf_FILTERS];

	L_Baises = new float[L_NumberOf_FILTERS];

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


	dim3 blockGrid(rowShifts, 1, 1);
	dim3 threads(columnShifts, 1, 1);

	size_t inputElementCount = m_InputMatrixHeight * m_InputMatrixWidth;
	size_t filterMatrixElementCount = m_FilterSize * m_FilterSize;
	size_t outputElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth;

	int inputByteCount = inputElementCount * sizeof(float);
	int filterByteCount = filterMatrixElementCount * sizeof(float);
	int outputByteCount = outputElementCount * sizeof(float);


	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Filter;
	float* d_Output;


	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount);
	cudaMalloc((void**)&d_Filter, filterByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);


	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_InputMatrix, inputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Filter, m_Filter, filterByteCount, cudaMemcpyHostToDevice);

	ConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_OutputMatrixWidth, m_InputMatrixWidth, m_FilterSize, m_FilterSize);
	cudaDeviceSynchronize();

	cudaMemcpy(m_OutputMatrix, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

}


void Convolution::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{
	m_BackpropInputMatrixHeight = backPassHeight; ///OK
	m_BackpropInputMatrixWidth = backPassWidth; ///OK

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight + 2; ///OK
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth + 2; ///OK

	int arrayLength = backPassHeight * backPassWidth; ///OK
	size_t inputSize = arrayLength * sizeof(float); ///OK

	m_BackPropInputMatrix = new float[arrayLength]; ///OK
	m_BackpropagationOutput = new float[m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth]; ///OK

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize); ///OK

	//Main backprop

	FlipFilter(); ///OK
	PadBackpropInput(); ///OK

	int rowShifts = m_BackpropOutputMatrixHeight;
	int columnShifts = m_BackpropOutputMatrixWidth;


	dim3 blockGrid(rowShifts, 1, 1); ///OK
	dim3 threads(columnShifts, 1, 1); ///OK

	size_t inputElementCount = m_PaddedInputHeight * m_PaddedInputWidth; ///OK
	size_t filterMatrixElementCount = m_FilterSize * m_FilterSize; ///OK
	size_t outputElementCount = m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth; ///OK

	int inputByteCount = inputElementCount * sizeof(float); ///OK
	int filterByteCount = filterMatrixElementCount * sizeof(float); ///OK
	int outputByteCount = outputElementCount * sizeof(float); ///OK

	//Define pointers for deviceMemory locations
	float* d_Input; ///OK
	float* d_Filter; ///OK
	float* d_Output; ///OK


	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount); ///OK
	cudaMalloc((void**)&d_Filter, filterByteCount); ///OK
	cudaMalloc((void**)&d_Output, outputByteCount); ///OK


	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_PaddedBackpropInput, inputByteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_Filter, m_FlippedFilter, filterByteCount, cudaMemcpyHostToDevice); ///OK

	ConvolutionKernel << <blockGrid, threads >> > (d_Input, d_Filter, d_Output, m_BackpropOutputMatrixWidth, m_PaddedInputWidth, m_FilterSize, m_FilterSize);
	cudaDeviceSynchronize(); ///OK


	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost); ///OK

	//Filter backprop
	FilterBackprop(backpropInput, backPassHeight, backPassWidth); ///OK


	cudaFree(d_Input); ///OK
	cudaFree(d_Filter); ///OK
	cudaFree(d_Output); ///OK
}

void Convolution::FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth)
{
	size_t fwdInputElementCount = m_InputMatrixHeight * m_InputMatrixWidth; ///OK
	size_t filterEqivElementCount = m_OutputMatrixHeight * m_OutputMatrixWidth; ///OK
	size_t  filterOutputElementCount = m_FilterSize * m_FilterSize; ///OK

	m_FilterBackpropResult = new float[filterOutputElementCount];

	int fwdInputByteCount = fwdInputElementCount * sizeof(float); ///OK
	int filterEqivByteCount = filterEqivElementCount * sizeof(float); ///OK
	int filterOutputByteCount = filterOutputElementCount * sizeof(float); ///OK

	float* d_FwdInput; ///OK
	float* d_FilterEquiv; ///OK
	float* d_FilterOutput; ///OK

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, fwdInputByteCount); ///OK
	cudaMalloc((void**)&d_FilterEquiv, filterEqivByteCount); ///OK
	cudaMalloc((void**)&d_FilterOutput, filterOutputByteCount); ///OK

	//Copy m_Filter into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_FwdInput, m_InputMatrix, fwdInputByteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_FilterEquiv, m_BackPropInputMatrix, filterEqivByteCount, cudaMemcpyHostToDevice); ///OK

	int rowShifts = m_FilterSize; ///OK
	int columnShifts = m_FilterSize; ///OK

	dim3 blockGrid(rowShifts, 1, 1); ///OK
	dim3 threads(columnShifts, 1, 1); ///OK

	ConvolutionKernel << <blockGrid, threads >> > (d_FwdInput, d_FilterEquiv, d_FilterOutput, m_FilterSize, m_InputMatrixWidth, m_OutputMatrixHeight, m_OutputMatrixWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(m_FilterBackpropResult, d_FilterOutput, filterOutputByteCount, cudaMemcpyDeviceToHost);
}


void Convolution::LayerForwardPass(float** _inputs)
{
	int inputSize = L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH;

	int filterSize = m_FilterSize * m_FilterSize;

	int outputSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t filterByteCount = filterSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);

	for (int inputNumber = 0; inputNumber < L_FORWARD_NumberOf_INPUTS; ++inputNumber)
	{
		L_FORWARD_Pass_INPUTS[inputNumber] = new float[inputSize];
		memcpy(L_FORWARD_Pass_INPUTS[inputNumber], _inputs[inputNumber], inputByteCount);
	}

	for (int outputNumber = 0; outputNumber < L_FORWARD_NumerOf_OUTPUTS; ++outputNumber)
	{
		L_FORWARD_Pass_OUTPUTS[outputNumber] = new float[outputSize];
	}

	int numberOfBlockx_X = L_FORWARD_OutputLayer_HEIGHT;
	int numberOfBlocks_Y = L_FORWARD_OutputLayer_WIDTH;
	int numberOfBlocks_Z = L_FORWARD_NumerOf_OUTPUTS;
	int numberOfThreadsPerBlock = L_FORWARD_NumberOf_INPUTS;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Inputs = new float* [L_FORWARD_NumberOf_INPUTS];  //(float**)malloc(L_FORWARD_NumberOf_INPUTS * sizeof(int*));
	float** h_Filters = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_Outputs = new float* [L_FORWARD_NumerOf_OUTPUTS]; //(float**)malloc(L_FORWARD_NumerOf_OUTPUTS * sizeof(int*));

	float* d_Biases;
	cudaMalloc((void**)&d_Biases, L_NumberOf_FILTERS * sizeof(float));


	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_FilterPointerArray;
	cudaMalloc((void**)&d_FilterPointerArray, L_NumberOf_FILTERS * sizeof(int*));

	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_FORWARD_NumerOf_OUTPUTS * sizeof(int*));


	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Inputs[i], inputByteCount);
		cudaMemcpy(h_Inputs[i], L_FORWARD_Pass_INPUTS[i], inputByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Filters[i], filterByteCount);
		cudaMemcpy(h_Filters[i], L_Filters[i], filterByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_FORWARD_NumerOf_OUTPUTS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
		cudaMemset(&h_Outputs[i], 0, outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_FORWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterPointerArray, h_Filters, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_FORWARD_NumerOf_OUTPUTS * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, numberOfBlocks_Y, numberOfBlocks_Z); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	LayerConvolutionKernal << <blockGrid, threads >> > (d_InputPointerArray, d_FilterPointerArray, d_OutputPointerArray, d_Biases, L_FORWARD_NumberOf_INPUTS, L_FORWARD_OutputLayer_WIDTH, L_FORWARD_InputLayer_WIDTH, m_FilterSize);
	cudaDeviceSynchronize();
	testIt = new float[outputSize];
	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_FORWARD_NumerOf_OUTPUTS * sizeof(int*), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < L_FORWARD_NumerOf_OUTPUTS; i++) {

		cudaMemcpy(testIt, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_FORWARD_Pass_OUTPUTS[i], testIt, outputByteCount);
		//cudaFree(h_Outputs[i]);
	}
	//cudaFree(d_OutputPointerArray);
	//delete[] h_Outputs;

	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaFree(&d_InputPointerArray[i]);
	}
	cudaFree(d_InputPointerArray);
	delete[] h_Inputs;

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaFree(&d_FilterPointerArray[i]);
	}
	cudaFree(d_FilterPointerArray);
	delete[] h_Filters;
}


void Convolution::LayerBackwardPass(float** _backpropInput)
{

}



void Convolution::PadBackpropInput()
{
	m_PaddedInputHeight = m_BackpropInputMatrixHeight + 2 * m_PaddingSize;
	m_PaddedInputWidth = m_BackpropInputMatrixWidth + 2 * m_PaddingSize;
	int elementsInPaddedInput = m_PaddedInputHeight * m_PaddedInputWidth;

	m_PaddedBackpropInput = new float[elementsInPaddedInput];

	memset(m_PaddedBackpropInput, 0, elementsInPaddedInput * sizeof(float));

	float* d_Output;
	float* d_UnpaddedInput;

	size_t outputByteCount = elementsInPaddedInput * sizeof(float);
	size_t unpaddedInputByteCount = (m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth) * sizeof(float);

	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMalloc((void**)&d_UnpaddedInput, unpaddedInputByteCount);

	cudaMemcpy(d_UnpaddedInput, m_BackPropInputMatrix, unpaddedInputByteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Output, m_PaddedBackpropInput, unpaddedInputByteCount, cudaMemcpyHostToDevice);

	int numberOfBlocks = m_BackpropInputMatrixHeight / 2;
	cout << "Launching blocks: " << numberOfBlocks << endl;
	dim3 blockGrid(numberOfBlocks, 1, 1);
	dim3 threads(2, 1, 1);

	ConvolutionPaddingKernel << <blockGrid, threads >> > (d_UnpaddedInput, d_Output, m_PaddedInputWidth, m_BackpropInputMatrixWidth, m_BackpropInputMatrixHeight);
	cudaDeviceSynchronize();

	cudaMemcpy(m_PaddedBackpropInput, d_Output, outputByteCount, cudaMemcpyDeviceToHost);
}

void Convolution::UpdateModule()
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


void Convolution::Dialate(float* _input, float* _output)
{

}

void Convolution::InitializeFilter()
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> distribution{ 1,2 };
	m_Filter = new float[m_FilterSize * m_FilterSize];

	for (int i = 0; i < m_FilterSize * m_FilterSize; ++i)
	{
		m_Filter[i] = distribution(gen);  //TODO FIX INITIALIZATION
	}
}

void Convolution::LayerFilterInitialization()
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> distribution{ 1,2 };

	int filterElementCount = m_FilterSize * m_FilterSize;

	for (int filterNumber = 0; filterNumber < L_NumberOf_FILTERS; ++filterNumber)
	{
		L_Filters[filterNumber] = new float[filterElementCount];

		for (int i = 0; i < filterElementCount; ++i)
		{
			L_Filters[filterNumber][i] = filterNumber; //distribution(gen);
		}
	}
}


void Convolution::FlipFilter()
{
	int filterArraySize = m_FilterSize * m_FilterSize;
	m_FlippedFilter = new float[filterArraySize];

	int k = 0;

	//Loop from back and assign value to new array
	for (int i = filterArraySize - 1; i >= 0; ) {
		m_FlippedFilter[k++] = m_Filter[i--];
	}
}


void Convolution::SetHyperParams(float _beta1, float _beta2, float _eps, int _t, float _alpha)
{
	m_HyperParam_Beta1 = _beta1;
	m_HyperParam_Beta2 = _beta2;
	m_HyperParam_Epsilon = _eps;
	m_HyperParam_T = _t;
	m_HyperParam_alpha = _alpha;
}


