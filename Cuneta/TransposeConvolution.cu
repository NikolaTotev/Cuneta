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

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif

#include <device_functions.h>
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

__global__ void LayerTransposeConvolutionKernel(float** _inputs, float** _filters, float** _outputs, float** _biases, int _numberOfOutputs, int _outputWidth, int _inputWidth, int _filterSize)
{
	int inputSelectionIndex = threadIdx.x;
	int filterSelectionIndex = blockIdx.z * _numberOfOutputs + threadIdx.x;
	int biasSelectionIndex = blockIdx.z * _numberOfOutputs + threadIdx.x;
	int outputSelectionIndex = blockIdx.z;

	float* selectedInput = _inputs[inputSelectionIndex];
	float* selectedFilter = _filters[filterSelectionIndex];
	float* selectedBias = _biases[filterSelectionIndex];
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

	result += selectedBias[outputArrayIndex];
	atomicAdd(&selectedOutput[outputArrayIndex], result);
};

__global__ void LayerTransposeConvolutionBackPropKernel(float** _inputs, float** _filters, float** _outputs, int _numberOfOutputs, int _outputWidth, int _inputWidth, int _filterSize)
{
	int inputSelectionIndex = threadIdx.x;
	int filterSelectionIndex = blockIdx.z * _numberOfOutputs + threadIdx.x;
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

	atomicAdd(&selectedOutput[outputArrayIndex], result);
}

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

__global__ void LayerTransposeInputPaddingKernel(float** __inputs, float** _outputs, int _unpaddedWidth, int _paddedWidth)
{
	int inputSelectionIndex = blockIdx.y;
	int outputSelectionIndex = blockIdx.y;

	float* selectedInput = __inputs[inputSelectionIndex];
	float* selectedOutput = _outputs[outputSelectionIndex];

	int rowWriteIndex = (blockIdx.x + 1) * 2 + threadIdx.x;
	int columnWriteIndex = 2;

	int inputRowReadIndex = (blockIdx.x * 2) + threadIdx.x;
	int inputColumnReadIndex = 0;

	int arrayPosition = rowWriteIndex * _paddedWidth + columnWriteIndex;
	int inputArrayPosition = inputRowReadIndex * _unpaddedWidth + inputColumnReadIndex;

	int var = 0;
	for (int i = 0; i < _unpaddedWidth; i++)
	{
		selectedOutput[arrayPosition] = selectedInput[inputArrayPosition];
		arrayPosition++;
		inputArrayPosition++;
	}
}

__global__ void LayerTransposeConvFilterFlipKernel(float** _inputFilters, float** _outputFilters, int _filterSize)
{
	float* filterToFlip = _inputFilters[blockIdx.x];
	int filterArraySize = _filterSize * _filterSize;
	float* flippedOutput = _outputFilters[blockIdx.x];
	int k = 0;

	//Loop from back and assign value to new array
	for (int i = filterArraySize - 1; i >= 0; ) {
		flippedOutput[k++] = filterToFlip[i--];
	}
}

__global__ void LayerTransposeConvFilterBackpropKernel(float** _inputs, float** _outputs, float** _filterEquivalents, int _inputsWidth, int _outputsWidth, int _filterEquivsHeight, int _filterEquivsWidth, int _numberOfInputs)
{
	int inputSelectionIndex = threadIdx.x;
	int filterSelectionIndex = blockIdx.z;
	int outputSelectionIndex = blockIdx.z * _numberOfInputs + threadIdx.x;

	float* selectedInput = _inputs[inputSelectionIndex];
	float* selectedFilter = _filterEquivalents[filterSelectionIndex];
	float* selectedOutput = _outputs[outputSelectionIndex];

	int inputStartReadRowIndex = blockIdx.x;
	int inputStartReadColumnIndex = blockIdx.y;

	int outputWriteRowIndex = blockIdx.x;
	int outputWriteColumnIndex = blockIdx.y;

	int inputArrayIndex = 0;

	int outputArrayIndex = outputWriteRowIndex * _outputsWidth + outputWriteColumnIndex;

	float result = 0;
	int filterIndex = 0;

	for (int row = 0; row < _filterEquivsHeight; row++)
	{
		inputStartReadColumnIndex = blockIdx.y;

		for (int col = 0; col < _filterEquivsWidth; col++)
		{
			inputArrayIndex = inputStartReadRowIndex * _inputsWidth + inputStartReadColumnIndex;

			result += selectedInput[inputArrayIndex] * selectedFilter[filterIndex];
			filterIndex++;
			inputStartReadColumnIndex += 1;
		}
		inputStartReadRowIndex += 1;
	}

	selectedOutput[outputArrayIndex] = result;
}

__global__ void TransposeConvFilterUpdateKernel(float** _currentFilters, float** _filterGradients, float** _VMatricies, float** _SMatricies, float** _V_CorrectedMatrices, float** _S_CorrectedMatricies, int _filterSize, int _HyperParam_Beta1, int _HyperParam_Beta2, int _HyperParam_T, int _HyperParam_alpha, int _HyperParam_Epsilon)
{
	float* selectedFilter = _currentFilters[blockIdx.x];
	float* selectedGradient = _filterGradients[blockIdx.x];
	float* selected_V_Matrix = _VMatricies[blockIdx.x];
	float* selected_S_Matrix = _SMatricies[blockIdx.x];
	float* selected_Corrected_V_Matrix = _V_CorrectedMatrices[blockIdx.x];
	float* selected_Corrected_S_Matrix = _S_CorrectedMatricies[blockIdx.x];

	for (int rowIndex = 0; rowIndex < _filterSize; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _filterSize; ++columnIndex)
		{
			int index = rowIndex * _filterSize + columnIndex;

			float filterBackpropValue = selectedGradient[index];
			float oldV = selected_V_Matrix[index];
			float oldS = selected_S_Matrix[index];

			float newV = _HyperParam_Beta1 * oldV + (1 - _HyperParam_Beta1) * filterBackpropValue;
			float newS = _HyperParam_Beta2 * oldS + (1 - _HyperParam_Beta2) * filterBackpropValue;

			float newVCorrected = newV / (1 - pow(_HyperParam_Beta1, _HyperParam_T));
			float newSCorrected = newS / (1 - pow(_HyperParam_Beta2, _HyperParam_T));

			selected_V_Matrix[index] = newV;
			selected_S_Matrix[index] = newS;

			selected_Corrected_V_Matrix[index] = newVCorrected;
			selected_Corrected_S_Matrix[index] = newSCorrected;

			float oldFilterValue = selectedFilter[index];
			float newF = oldFilterValue - _HyperParam_alpha * (newVCorrected / sqrt(newSCorrected + _HyperParam_Epsilon));

			selectedFilter[index] = newF;
		}
	}
}

__global__ void TransposeConvBiasUpdateKernel(float** _currentFilters, float** _filterGradients, float** _VMatricies, float** _SMatricies, float** _V_CorrectedMatrices, float** _S_CorrectedMatricies, int _height, int _width, int _HyperParam_Beta1, int _HyperParam_Beta2, int _HyperParam_T, int _HyperParam_alpha, int _HyperParam_Epsilon)
{
	float* selectedFilter = _currentFilters[blockIdx.x];
	float* selectedGradient = _filterGradients[blockIdx.x];
	float* selected_V_Matrix = _VMatricies[blockIdx.x];
	float* selected_S_Matrix = _SMatricies[blockIdx.x];
	float* selected_Corrected_V_Matrix = _V_CorrectedMatrices[blockIdx.x];
	float* selected_Corrected_S_Matrix = _S_CorrectedMatricies[blockIdx.x];

	for (int rowIndex = 0; rowIndex < _height; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _width; ++columnIndex)
		{
			int index = rowIndex * _width + columnIndex;

			float filterBackpropValue = selectedGradient[index];
			float oldV = selected_V_Matrix[index];
			float oldS = selected_S_Matrix[index];

			float newV = _HyperParam_Beta1 * oldV + (1 - _HyperParam_Beta1) * filterBackpropValue;
			float newS = _HyperParam_Beta2 * oldS + (1 - _HyperParam_Beta2) * filterBackpropValue;

			float newVCorrected = newV / (1 - pow(_HyperParam_Beta1, _HyperParam_T));
			float newSCorrected = newS / (1 - pow(_HyperParam_Beta2, _HyperParam_T));

			selected_V_Matrix[index] = newV;
			selected_S_Matrix[index] = newS;

			selected_Corrected_V_Matrix[index] = newVCorrected;
			selected_Corrected_S_Matrix[index] = newSCorrected;

			float oldFilterValue = selectedFilter[index];
			float newF = oldFilterValue - _HyperParam_alpha * (newVCorrected / sqrt(newSCorrected + _HyperParam_Epsilon));

			selectedFilter[index] = newF;
		}
	}
}


TransposeConvolution::TransposeConvolution(int _filterSize, int _paddingSize, int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth)
{
	m_FilterSize = _filterSize;
	m_PaddingSize = _paddingSize;

	L_FORWARD_NumberOf_INPUTS = _numberOfInputs;
	L_FORWARD_NumberOf_OUTPUTS = _numberOfOutputs;

	L_BACKWARD_NumberOf_INPUTS = L_FORWARD_NumberOf_OUTPUTS;
	L_BACKWARD_NumberOf_OUTPUTS = L_FORWARD_NumberOf_INPUTS;

	L_FORWARD_InputLayer_HEIGHT = _inputHeight;
	L_FORWARD_InputLayer_WIDTH = _inputWidth;

	L_FORWARD_OutputLayer_HEIGHT = _inputHeight  +2;
	L_FORWARD_OutputLayer_WIDTH = _inputWidth  +2;

	L_BACKWARD_InputLayer_HEIGHT = L_FORWARD_OutputLayer_HEIGHT;
	L_BACKWARD_InputLayer_WIDTH = L_FORWARD_OutputLayer_WIDTH;

	L_BACKWARD_OutputLayer_HEIGHT = L_FORWARD_InputLayer_HEIGHT;
	L_BACKWARD_OutputLayer_WIDTH = L_FORWARD_InputLayer_WIDTH;

	L_FORWARD_Pass_INPUTS = new float* [L_FORWARD_NumberOf_INPUTS];
	L_FORWARD_Pass_PADDED_INPUTS = new float* [L_FORWARD_NumberOf_INPUTS];
	L_FORWARD_Pass_OUTPUTS = new float* [L_FORWARD_NumberOf_OUTPUTS];

	L_BACKWARD_Pass_INPUTS = new float* [L_BACKWARD_NumberOf_INPUTS];
	L_BACKWARD_Pass_OUTPUTS = new float* [L_BACKWARD_NumberOf_OUTPUTS];

	L_NumberOf_FILTERS = L_FORWARD_NumberOf_INPUTS * L_FORWARD_NumberOf_OUTPUTS;

	L_Filters = new float* [L_NumberOf_FILTERS];
	L_FLIPPED_Filters = new float* [L_NumberOf_FILTERS];
	L_Filter_BACKPROP_RESULTS = new float* [L_NumberOf_FILTERS];

	L_Baises = new float*[L_NumberOf_FILTERS];
	L_PrevBiases = new float* [L_NumberOf_FILTERS];

	L_AdamOptimizer_V_Matrix = new float* [L_NumberOf_FILTERS];
	L_AdamOptimizer_S_Matrix = new float* [L_NumberOf_FILTERS];
	L_AdamOptimizer_Corrected_V_Matrix = new float* [L_NumberOf_FILTERS];
	L_AdamOptimizer_Corrected_S_Matrix = new float* [L_NumberOf_FILTERS];

	L_BIAS_AdamOptimizer_V_Matrix = new float* [L_NumberOf_FILTERS];
	L_BIAS_AdamOptimizer_S_Matrix = new float* [L_NumberOf_FILTERS];
	L_BIAS_AdamOptimizer_Corrected_V_Matrix = new float* [L_NumberOf_FILTERS];
	L_BIAS_AdamOptimizer_Corrected_S_Matrix = new float* [L_NumberOf_FILTERS];
	

	for (int i = 0; i < L_NumberOf_FILTERS; ++i)
	{
		L_AdamOptimizer_V_Matrix[i] = new float[m_FilterSize * m_FilterSize];
		L_AdamOptimizer_S_Matrix[i] = new float[m_FilterSize * m_FilterSize];
		L_AdamOptimizer_Corrected_V_Matrix[i] = new float[m_FilterSize * m_FilterSize];
		L_AdamOptimizer_Corrected_S_Matrix[i] = new float[m_FilterSize * m_FilterSize];

		size_t byteCount = m_FilterSize * m_FilterSize * sizeof(float);
		memset(L_AdamOptimizer_V_Matrix[i], 0, byteCount);
		memset(L_AdamOptimizer_S_Matrix[i], 0, byteCount);
		memset(L_AdamOptimizer_Corrected_V_Matrix[i], 0, byteCount);
		memset(L_AdamOptimizer_Corrected_S_Matrix[i], 0, byteCount);

		L_Biases = new float* [L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH];
		L_BIAS_AdamOptimizer_V_Matrix[i] = new float[L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH];
		L_BIAS_AdamOptimizer_S_Matrix[i] = new float[L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH];
		L_BIAS_AdamOptimizer_Corrected_V_Matrix[i] = new float[L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH];
		L_BIAS_AdamOptimizer_Corrected_S_Matrix[i] = new float[L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH];

		size_t biasByteCount = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH * sizeof(float);
		memset(L_BIAS_AdamOptimizer_V_Matrix[i], 0, biasByteCount);
		memset(L_BIAS_AdamOptimizer_S_Matrix[i], 0, biasByteCount);
		memset(L_BIAS_AdamOptimizer_Corrected_V_Matrix[i], 0, biasByteCount);
		memset(L_BIAS_AdamOptimizer_Corrected_S_Matrix[i], 0, biasByteCount);
	}

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

void TransposeConvolution::LayerForwardPass(float** _inputs)
{
	int inputSize = L_FORWARD_InputLayer_HEIGHT* L_FORWARD_InputLayer_WIDTH;
	
	int filterSize = m_FilterSize * m_FilterSize;

	int biasSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

	int outputSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t filterByteCount = filterSize * sizeof(float);
	size_t biasByteCount = biasSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);

	for (int inputNumber = 0; inputNumber < L_FORWARD_NumberOf_INPUTS; ++inputNumber)
	{
		L_FORWARD_Pass_INPUTS[inputNumber] = new float[inputSize];
		memcpy(L_FORWARD_Pass_INPUTS[inputNumber], _inputs[inputNumber], inputByteCount);
	}
	
	for (int outputNumber = 0; outputNumber < L_FORWARD_NumberOf_OUTPUTS; ++outputNumber)
	{
		L_FORWARD_Pass_OUTPUTS[outputNumber] = new float[outputSize];
	}

	LayerPadInput();

	int paddedInputSize = L_FORWARD_InputLayer_PADDED_HEIGHT * L_FORWARD_InputLayer_PADDED_WIDTH;
	size_t paddedInputByteCount = paddedInputSize * sizeof(float);


	int numberOfBlockx_X = L_FORWARD_OutputLayer_HEIGHT;//L_FORWARD_OutputLayer_HEIGHT;
	int numberOfBlocks_Y = L_FORWARD_OutputLayer_WIDTH;//L_FORWARD_OutputLayer_WIDTH;
	int numberOfBlocks_Z = L_FORWARD_NumberOf_OUTPUTS;//L_FORWARD_NumberOf_OUTPUTS;
	int numberOfThreadsPerBlock = L_FORWARD_NumberOf_INPUTS;//L_FORWARD_NumberOf_INPUTS;

	float** h_Inputs = new float* [L_FORWARD_NumberOf_INPUTS];  //(float**)malloc(L_FORWARD_NumberOf_INPUTS * sizeof(int*));
	float** h_Filters = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_Biases = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_Outputs = new float* [L_FORWARD_NumberOf_OUTPUTS]; //(float**)malloc(L_FORWARD_NumberOf_OUTPUTS * sizeof(int*));

	float** d_BiasesPointerArray;
	cudaMalloc((void**)&d_BiasesPointerArray, L_NumberOf_FILTERS * sizeof(int*));

	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_FilterPointerArray;
	cudaMalloc((void**)&d_FilterPointerArray, L_NumberOf_FILTERS * sizeof(int*));

	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_FORWARD_NumberOf_OUTPUTS * sizeof(int*));

	float* something = new float[paddedInputSize];
	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Inputs[i], paddedInputByteCount);
		memcpy(something, L_FORWARD_Pass_PADDED_INPUTS[i], paddedInputByteCount);
		cudaMemcpy(h_Inputs[i], L_FORWARD_Pass_PADDED_INPUTS[i], paddedInputByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Filters[i], filterByteCount);
		cudaMemcpy(h_Filters[i], L_Filters[i], filterByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Biases[i], filterByteCount);
		cudaMemcpy(h_Biases[i], L_Biases[i], biasByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_FORWARD_NumberOf_OUTPUTS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
		cudaMemset(&h_Outputs[i], 0, outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_FORWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterPointerArray, h_Filters, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_BiasesPointerArray, h_Biases, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);

	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_FORWARD_NumberOf_OUTPUTS * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, numberOfBlocks_Y, numberOfBlocks_Z); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	LayerTransposeConvolutionKernel << <blockGrid, threads >> > (d_InputPointerArray, d_FilterPointerArray, d_OutputPointerArray,d_BiasesPointerArray , L_FORWARD_NumberOf_OUTPUTS, L_FORWARD_OutputLayer_WIDTH, L_FORWARD_InputLayer_PADDED_WIDTH, m_FilterSize);
	cudaDeviceSynchronize();
	float* temp = new float[outputSize];
	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_FORWARD_NumberOf_OUTPUTS * sizeof(int*), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < L_FORWARD_NumberOf_OUTPUTS; i++) {

		cudaMemcpy(temp, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_FORWARD_Pass_OUTPUTS[i], temp, outputByteCount);
		cudaFree(h_Outputs[i]);
	}
	cudaFree(d_OutputPointerArray);
	delete[] h_Outputs;

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

void TransposeConvolution::LayerBackwardPass(float** _backpropInput)
{
	int inputSize = L_BACKWARD_InputLayer_HEIGHT * L_BACKWARD_InputLayer_WIDTH;

	int filterSize = m_FilterSize * m_FilterSize;

	int outputSize = L_BACKWARD_OutputLayer_HEIGHT * L_BACKWARD_OutputLayer_WIDTH;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t filterByteCount = filterSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);

	for (int inputNumber = 0; inputNumber < L_BACKWARD_NumberOf_INPUTS; ++inputNumber)
	{
		L_BACKWARD_Pass_INPUTS[inputNumber] = new float[inputSize];
		memcpy(L_BACKWARD_Pass_INPUTS[inputNumber], _backpropInput[inputNumber], inputByteCount);
	}

	for (int outputNumber = 0; outputNumber < L_BACKWARD_NumberOf_OUTPUTS; ++outputNumber)
	{
		L_BACKWARD_Pass_OUTPUTS[outputNumber] = new float[outputSize];
	}

	LayerFlipFilter();

	int numberOfBlockx_X = L_BACKWARD_OutputLayer_HEIGHT;
	int numberOfBlocks_Y = L_BACKWARD_OutputLayer_WIDTH;
	int numberOfBlocks_Z = L_BACKWARD_NumberOf_OUTPUTS;
	int numberOfThreadsPerBlock = L_BACKWARD_NumberOf_INPUTS;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Inputs = new float* [L_BACKWARD_NumberOf_INPUTS];  //(float**)malloc(L_FORWARD_NumberOf_INPUTS * sizeof(int*));
	float** h_Filters = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_Outputs = new float* [L_BACKWARD_NumberOf_OUTPUTS]; //(float**)malloc(L_FORWARD_NumberOf_OUTPUTS * sizeof(int*));

	float* d_Biases;
	cudaMalloc((void**)&d_Biases, L_NumberOf_FILTERS * sizeof(float));


	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_BACKWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_FilterPointerArray;
	cudaMalloc((void**)&d_FilterPointerArray, L_NumberOf_FILTERS * sizeof(int*));

	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_BACKWARD_NumberOf_OUTPUTS * sizeof(int*));


	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_BACKWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Inputs[i], inputByteCount);
		cudaMemcpy(h_Inputs[i], L_BACKWARD_Pass_INPUTS[i], inputByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Filters[i], filterByteCount);
		cudaMemcpy(h_Filters[i], L_FLIPPED_Filters[i], filterByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_BACKWARD_NumberOf_OUTPUTS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
		cudaMemset(&h_Outputs[i], 0, outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_BACKWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterPointerArray, h_Filters, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_BACKWARD_NumberOf_OUTPUTS * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, numberOfBlocks_Y, numberOfBlocks_Z); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	LayerTransposeConvolutionBackPropKernel << <blockGrid, threads >> > (d_InputPointerArray, d_FilterPointerArray, d_OutputPointerArray, L_BACKWARD_NumberOf_INPUTS, L_BACKWARD_OutputLayer_WIDTH, L_BACKWARD_InputLayer_WIDTH, m_FilterSize);
	cudaDeviceSynchronize();
	float* temp = new float[outputSize];
	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_BACKWARD_NumberOf_OUTPUTS * sizeof(int*), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < L_BACKWARD_NumberOf_OUTPUTS; i++) {

		cudaMemcpy(temp, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_BACKWARD_Pass_OUTPUTS[i], temp, outputByteCount);
		cudaFree(h_Outputs[i]);
	}
	cudaFree(d_OutputPointerArray);
	delete[] h_Outputs;

	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_BACKWARD_NumberOf_INPUTS; i++) {
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

void TransposeConvolution::LayerFilterBackprop()
{
	int inputSize = L_FORWARD_InputLayer_PADDED_WIDTH* L_FORWARD_InputLayer_PADDED_WIDTH;

	int filterEquivalentSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

	int outputSize = m_FilterSize * m_FilterSize;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t filterEquivalentByteCount = filterEquivalentSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);


	for (int outputNumber = 0; outputNumber < L_NumberOf_FILTERS; ++outputNumber)
	{
		L_Filter_BACKPROP_RESULTS[outputNumber] = new float[outputSize];
	}


	int numberOfBlockx_X = m_FilterSize;
	int numberOfBlocks_Y = m_FilterSize;
	int numberOfBlocks_Z = L_BACKWARD_NumberOf_INPUTS;
	int numberOfThreadsPerBlock = L_FORWARD_NumberOf_INPUTS;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Inputs = new float* [L_FORWARD_NumberOf_INPUTS];  //(float**)malloc(L_FORWARD_NumberOf_INPUTS * sizeof(int*));
	float** h_Filters = new float* [L_BACKWARD_NumberOf_INPUTS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_Outputs = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_FORWARD_NumberOf_OUTPUTS * sizeof(int*));



	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_FilterPointerArray;
	cudaMalloc((void**)&d_FilterPointerArray, L_BACKWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_NumberOf_FILTERS * sizeof(int*));


	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Inputs[i], inputByteCount);
		cudaMemcpy(h_Inputs[i], L_FORWARD_Pass_PADDED_INPUTS[i], inputByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_BACKWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Filters[i], filterEquivalentByteCount);
		cudaMemcpy(h_Filters[i], L_BACKWARD_Pass_INPUTS[i], filterEquivalentByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
		cudaMemset(&h_Outputs[i], 0, outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_FORWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterPointerArray, h_Filters, L_BACKWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_NumberOf_FILTERS * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, numberOfBlocks_Y, numberOfBlocks_Z); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	LayerTransposeConvFilterBackpropKernel << <blockGrid, threads >> > (d_InputPointerArray, d_OutputPointerArray, d_FilterPointerArray, L_FORWARD_InputLayer_PADDED_WIDTH, m_FilterSize, L_BACKWARD_InputLayer_HEIGHT, L_BACKWARD_InputLayer_WIDTH, L_FORWARD_NumberOf_INPUTS);
	cudaDeviceSynchronize();

	float* temp = new float[outputSize];
	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {

		cudaMemcpy(temp, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_Filter_BACKPROP_RESULTS[i], temp, outputByteCount);
		cudaFree(h_Outputs[i]);
	}
	cudaFree(d_OutputPointerArray);
	delete[] h_Outputs;

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

void TransposeConvolution::LayerFlipFilter()
{
	int inputSize = m_FilterSize * m_FilterSize;

	int outputSize = m_FilterSize * m_FilterSize;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);

	for (int filterNumber = 0; filterNumber < L_NumberOf_FILTERS; ++filterNumber)
	{
		L_FLIPPED_Filters[filterNumber] = new float[inputSize];
	}

	int numberOfBlockx_X = L_NumberOf_FILTERS;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Inputs = new float* [L_NumberOf_FILTERS];

	float** h_Outputs = new float* [L_NumberOf_FILTERS];

	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_NumberOf_FILTERS * sizeof(int*));


	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_NumberOf_FILTERS * sizeof(int*));


	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Inputs[i], inputByteCount);
		cudaMemcpy(h_Inputs[i], L_Filters[i], inputByteCount, cudaMemcpyHostToDevice);
	}

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, 1, 1); ///OK
	dim3 threads(1, 1, 1); ///OK

	LayerTransposeConvFilterFlipKernel << <blockGrid, threads >> > (d_InputPointerArray, d_OutputPointerArray, m_FilterSize);
	cudaDeviceSynchronize();

	float* temp = new float[outputSize];

	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {

		cudaMemcpy(temp, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_FLIPPED_Filters[i], temp, outputByteCount);
		cudaFree(&h_Outputs[i]);
	}
	cudaFree(d_OutputPointerArray);
	delete[] h_Outputs;

	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaFree(&d_InputPointerArray[i]);
	}
	cudaFree(d_InputPointerArray);
	delete[] h_Inputs;

	delete[] temp;
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

void TransposeConvolution::LayerUpdate()
{
	int filterSize = m_FilterSize * m_FilterSize;

	size_t filterByteCount = filterSize * sizeof(float);


	int numberOfBlocks_X = L_NumberOf_FILTERS;
	int numberOfThreadsPerBlock = 1;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Filters = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_FilterGradients = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_V_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_S_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_V_CORRECTED_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_S_CORRECTED_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));


	float** d_FilterPointers;
	cudaMalloc((void**)&d_FilterPointers, L_NumberOf_FILTERS * sizeof(int*));

	float** d_FilterGradientPointers;
	cudaMalloc((void**)&d_FilterGradientPointers, L_NumberOf_FILTERS * sizeof(int*));

	float** d_V_Matricies;
	cudaMalloc((void**)&d_V_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_S_Matricies;
	cudaMalloc((void**)&d_S_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_V_CORRECTED_Matricies;
	cudaMalloc((void**)&d_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_S_CORRECTED_Matricies;
	cudaMalloc((void**)&d_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*));



	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Filters[i], filterByteCount);
		cudaMemcpy(h_Filters[i], L_Filters[i], filterByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_FilterGradients[i], filterByteCount);
		cudaMemcpy(h_FilterGradients[i], L_Filter_BACKPROP_RESULTS[i], filterByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_V_Matricies[i], filterByteCount);
		cudaMemcpy(h_V_Matricies[i], L_AdamOptimizer_V_Matrix[i], filterByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_S_Matricies[i], filterByteCount);
		cudaMemcpy(h_S_Matricies[i], L_AdamOptimizer_S_Matrix[i], filterByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_V_CORRECTED_Matricies[i], filterByteCount);
		cudaMemcpy(h_V_CORRECTED_Matricies[i], L_AdamOptimizer_Corrected_V_Matrix[i], filterByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_S_CORRECTED_Matricies[i], filterByteCount);
		cudaMemcpy(h_S_CORRECTED_Matricies[i], L_AdamOptimizer_Corrected_S_Matrix[i], filterByteCount, cudaMemcpyHostToDevice);
	}


	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_FilterPointers, h_Filters, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FilterGradientPointers, h_FilterGradients, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V_Matricies, h_V_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_Matricies, h_S_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V_CORRECTED_Matricies, h_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_CORRECTED_Matricies, h_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);


	dim3 blockGrid(numberOfBlocks_X, 1, 1); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	TransposeConvFilterUpdateKernel << <blockGrid, threads >> > (d_FilterPointers, d_FilterGradientPointers, d_V_Matricies, d_S_Matricies, d_V_CORRECTED_Matricies, d_S_CORRECTED_Matricies, m_FilterSize, m_HyperParam_Beta1, m_HyperParam_Beta2, m_HyperParam_T, m_HyperParam_alpha, m_HyperParam_Epsilon);
	cudaDeviceSynchronize();

	float* temp = new float[filterByteCount];

	cudaMemcpy(h_Filters, d_FilterPointers, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_FilterGradients, d_FilterGradientPointers, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V_Matricies, d_V_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_Matricies, d_S_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V_CORRECTED_Matricies, h_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_CORRECTED_Matricies, h_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {

		cudaMemcpy(temp, h_Filters[i], filterByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_Filters[i], temp, filterByteCount);
		cudaFree(h_Filters[i]);

		cudaMemcpy(temp, h_V_Matricies[i], filterByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_AdamOptimizer_V_Matrix[i], temp, filterByteCount);
		cudaFree(h_V_Matricies[i]);

		cudaMemcpy(temp, h_S_Matricies[i], filterByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_AdamOptimizer_S_Matrix[i], temp, filterByteCount);
		cudaFree(h_S_Matricies[i]);

		cudaMemcpy(temp, h_V_CORRECTED_Matricies[i], filterByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_AdamOptimizer_Corrected_V_Matrix[i], temp, filterByteCount);
		cudaFree(h_V_CORRECTED_Matricies[i]);

		cudaMemcpy(temp, h_S_CORRECTED_Matricies[i], filterByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_AdamOptimizer_Corrected_S_Matrix[i], temp, filterByteCount);
		cudaFree(h_S_CORRECTED_Matricies[i]);
	}

	cudaFree(d_FilterPointers);
	cudaFree(d_FilterGradientPointers);
	cudaFree(d_V_Matricies);
	cudaFree(d_S_Matricies);
	cudaFree(h_V_CORRECTED_Matricies);
	cudaFree(h_S_CORRECTED_Matricies);

	delete[] h_Filters;
	delete[] h_FilterGradients;
	delete[] h_V_Matricies;
	delete[] h_S_Matricies;
	delete[] h_V_CORRECTED_Matricies;
	delete[] h_S_CORRECTED_Matricies;
}

void TransposeConvolution::LayerBiasUpdate()
{
	int biasMatrixSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

	size_t biasMatrixByteCount = biasMatrixSize * sizeof(float);


	int numberOfBlocks_X = L_NumberOf_FILTERS;
	int numberOfThreadsPerBlock = 1;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Biases = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_BiasGradients = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_V_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_S_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_V_CORRECTED_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));
	float** h_S_CORRECTED_Matricies = new float* [L_NumberOf_FILTERS]; //(float**)malloc(L_NumberOf_FILTERS * sizeof(int*));


	float** d_BiasPointers;
	cudaMalloc((void**)&d_BiasPointers, L_NumberOf_FILTERS * sizeof(int*));

	float** d_BiasGradientPointers;
	cudaMalloc((void**)&d_BiasGradientPointers, L_NumberOf_FILTERS * sizeof(int*));

	float** d_V_Matricies;
	cudaMalloc((void**)&d_V_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_S_Matricies;
	cudaMalloc((void**)&d_S_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_V_CORRECTED_Matricies;
	cudaMalloc((void**)&d_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*));

	float** d_S_CORRECTED_Matricies;
	cudaMalloc((void**)&d_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*));



	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {
		cudaMalloc(&h_Biases[i], biasMatrixByteCount);
		cudaMemcpy(h_Biases[i], L_Biases[i], biasMatrixByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_BiasGradients[i], biasMatrixByteCount);
		cudaMemcpy(h_BiasGradients[i], L_BACKWARD_Pass_INPUTS[i], biasMatrixByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_V_Matricies[i], biasMatrixByteCount);
		cudaMemcpy(h_V_Matricies[i], L_BIAS_AdamOptimizer_V_Matrix[i], biasMatrixByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_S_Matricies[i], biasMatrixByteCount);
		cudaMemcpy(h_S_Matricies[i], L_BIAS_AdamOptimizer_S_Matrix[i], biasMatrixByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_V_CORRECTED_Matricies[i], biasMatrixByteCount);
		cudaMemcpy(h_V_CORRECTED_Matricies[i], L_BIAS_AdamOptimizer_Corrected_V_Matrix[i], biasMatrixByteCount, cudaMemcpyHostToDevice);

		cudaMalloc(&h_S_CORRECTED_Matricies[i], biasMatrixByteCount);
		cudaMemcpy(h_S_CORRECTED_Matricies[i], L_BIAS_AdamOptimizer_Corrected_S_Matrix[i], biasMatrixByteCount, cudaMemcpyHostToDevice);
	}


	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_BiasPointers, h_Biases, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_BiasGradientPointers, h_BiasGradients, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V_Matricies, h_V_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_Matricies, h_S_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V_CORRECTED_Matricies, h_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_CORRECTED_Matricies, h_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyHostToDevice);


	dim3 blockGrid(numberOfBlocks_X, 1, 1); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	TransposeConvBiasUpdateKernel << <blockGrid, threads >> > (d_BiasPointers, d_BiasGradientPointers, d_V_Matricies, d_S_Matricies, d_V_CORRECTED_Matricies, d_S_CORRECTED_Matricies, L_FORWARD_OutputLayer_HEIGHT, L_FORWARD_OutputLayer_WIDTH, m_HyperParam_Beta1, m_HyperParam_Beta2, m_HyperParam_T, m_HyperParam_alpha, m_HyperParam_Epsilon);
	cudaDeviceSynchronize();

	float* temp = new float[biasMatrixByteCount];

	cudaMemcpy(h_Biases, d_BiasPointers, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_BiasGradients, d_BiasGradientPointers, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V_Matricies, d_V_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_Matricies, d_S_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V_CORRECTED_Matricies, h_V_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_CORRECTED_Matricies, h_S_CORRECTED_Matricies, L_NumberOf_FILTERS * sizeof(int*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < L_NumberOf_FILTERS; i++) {

		cudaMemcpy(temp, h_Biases[i], biasMatrixByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_Biases[i], temp, biasMatrixByteCount);
		cudaFree(h_Biases[i]);

		cudaMemcpy(temp, h_V_Matricies[i], biasMatrixByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_BIAS_AdamOptimizer_V_Matrix[i], temp, biasMatrixByteCount);
		cudaFree(h_V_Matricies[i]);

		cudaMemcpy(temp, h_S_Matricies[i], biasMatrixByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_BIAS_AdamOptimizer_S_Matrix[i], temp, biasMatrixByteCount);
		cudaFree(h_S_Matricies[i]);

		cudaMemcpy(temp, h_V_CORRECTED_Matricies[i], biasMatrixByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_BIAS_AdamOptimizer_Corrected_V_Matrix[i], temp, biasMatrixByteCount);
		cudaFree(h_V_CORRECTED_Matricies[i]);

		cudaMemcpy(temp, h_S_CORRECTED_Matricies[i], biasMatrixByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_BIAS_AdamOptimizer_Corrected_S_Matrix[i], temp, biasMatrixByteCount);
		cudaFree(h_S_CORRECTED_Matricies[i]);
	}

	cudaFree(d_BiasPointers);
	cudaFree(d_BiasGradientPointers);
	cudaFree(d_V_Matricies);
	cudaFree(d_S_Matricies);
	cudaFree(h_V_CORRECTED_Matricies);
	cudaFree(h_S_CORRECTED_Matricies);

	delete[] h_Biases;
	delete[] h_BiasGradients;
	delete[] h_V_Matricies;
	delete[] h_S_Matricies;
	delete[] h_V_CORRECTED_Matricies;
	delete[] h_S_CORRECTED_Matricies;
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

void TransposeConvolution::LayerPadInput()
{
	L_FORWARD_InputLayer_PADDED_HEIGHT = L_FORWARD_InputLayer_HEIGHT + 2 * m_PaddingSize;
	L_FORWARD_InputLayer_PADDED_WIDTH = L_FORWARD_InputLayer_WIDTH + 2 * m_PaddingSize;

	int inputSize = L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH;

	int outputSize = L_FORWARD_InputLayer_PADDED_HEIGHT * L_FORWARD_InputLayer_PADDED_WIDTH;

	size_t inputByteCount = inputSize * sizeof(float);
	size_t outputByteCount = outputSize * sizeof(float);


	for (int inputNumber = 0; inputNumber < L_FORWARD_NumberOf_INPUTS; ++inputNumber)
	{
		L_FORWARD_Pass_PADDED_INPUTS[inputNumber] = new float[outputSize];
	}

	int numberOfBlockx_X = L_FORWARD_InputLayer_HEIGHT / 2;
	int numberOfBlocks_Y = L_FORWARD_NumberOf_INPUTS;
	int numberOfThreadsPerBlock = 2;

	// create intermediate host array for storage of device row-pointers

	// create top-level device array pointer
	float** h_Inputs = new float* [L_FORWARD_NumberOf_INPUTS];  //(float**)malloc(L_FORWARD_NumberOf_INPUTS * sizeof(int*));
	float** h_Outputs = new float* [L_FORWARD_NumberOf_INPUTS]; //(float**)malloc(L_FORWARD_NumberOf_OUTPUTS * sizeof(int*));


	float** d_InputPointerArray;
	cudaMalloc((void**)&d_InputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*));

	float** d_OutputPointerArray;
	cudaMalloc((void**)&d_OutputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*));


	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Inputs[i], inputByteCount);
		cudaMemcpy(h_Inputs[i], L_FORWARD_Pass_INPUTS[i], inputByteCount, cudaMemcpyHostToDevice);
	}


	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {
		cudaMalloc(&h_Outputs[i], outputByteCount);
		cudaMemset(&h_Outputs[i], 0, outputByteCount);
	}

	// fixup top level device array pointer to point to array of device row-pointers
	cudaMemcpy(d_InputPointerArray, h_Inputs, L_FORWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OutputPointerArray, h_Outputs, L_FORWARD_NumberOf_INPUTS * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 blockGrid(numberOfBlockx_X, numberOfBlocks_Y, 1); ///OK
	dim3 threads(numberOfThreadsPerBlock, 1, 1); ///OK

	LayerTransposeInputPaddingKernel << <blockGrid, threads >> > (d_InputPointerArray, d_OutputPointerArray, L_FORWARD_InputLayer_WIDTH, L_FORWARD_InputLayer_PADDED_WIDTH);
	cudaDeviceSynchronize();

	float* temp = new float[outputSize];
	cudaMemcpy(h_Outputs, d_OutputPointerArray, L_FORWARD_NumberOf_INPUTS * sizeof(int*), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < L_FORWARD_NumberOf_INPUTS; i++) {

		cudaMemcpy(temp, h_Outputs[i], outputByteCount, cudaMemcpyDeviceToHost);
		memcpy(L_FORWARD_Pass_PADDED_INPUTS[i], temp, outputByteCount);
		cudaFree(h_Outputs[i]);
	}
	cudaFree(d_OutputPointerArray);
	delete[] h_Outputs;

	// allocate each device row-pointer, then copy host data to it
	for (size_t i = 0; i < L_BACKWARD_NumberOf_INPUTS; i++) {
		cudaFree(h_Inputs[i]);
	}
	cudaFree(d_InputPointerArray);
	delete h_Inputs;
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

void TransposeConvolution::LayerFilterInitialization()
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
			L_Filters[filterNumber][i] = i + 1; //distribution(gen);
		}
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

void TransposeConvolution::DebugPrintAll()
{
	int newLineCounter = 1;

	cout << "========================================================" << endl;
	cout << "============ Transpose Conv Debug Print All ============" << endl;
	cout << "========================================================" << endl;

	cout << "Squishy: " << endl;
	cout << "Layer ID: " << layerID << endl;
	cout << "Level ID: " << levelID << endl;
	cout << "Hyper parameters: " << endl;
	cout << "Beta 1: " << m_HyperParam_Beta1 << endl;
	cout << "Beta 2: " << m_HyperParam_Beta2 << endl;
	cout << "Epsilon: " << m_HyperParam_Epsilon << endl;
	cout << "Alpha: " << m_HyperParam_alpha << endl;
	cout << "T: " << m_HyperParam_T << endl;


	cout << ">>>> Forward Inputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_FORWARD_NumberOf_INPUTS; ++inputIndex)
	{
		cout << "--- Element " << inputIndex + 1 << "---" << endl;
		for (int elementIndex = 0; elementIndex < L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH; ++elementIndex)
		{
			cout << L_FORWARD_Pass_INPUTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_FORWARD_InputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Normal Filter Inputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_Filters[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}


	cout << ">>>> Flipped Filter Inputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_FLIPPED_Filters[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}


	cout << ">>>> Forward Outputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_FORWARD_NumberOf_OUTPUTS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH; ++elementIndex)
		{
			cout << L_FORWARD_Pass_OUTPUTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_FORWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Backward Inputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_BACKWARD_NumberOf_INPUTS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_BACKWARD_InputLayer_HEIGHT * L_BACKWARD_InputLayer_WIDTH; ++elementIndex)
		{
			cout << L_BACKWARD_Pass_INPUTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_BACKWARD_InputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Padded Backward Inputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_BACKWARD_NumberOf_INPUTS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_FORWARD_InputLayer_PADDED_HEIGHT * L_FORWARD_InputLayer_PADDED_WIDTH; ++elementIndex)
		{
			cout << L_FORWARD_Pass_PADDED_INPUTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_FORWARD_InputLayer_PADDED_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Backward Outputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_BACKWARD_NumberOf_OUTPUTS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_BACKWARD_OutputLayer_HEIGHT * L_BACKWARD_OutputLayer_WIDTH; ++elementIndex)
		{
			cout << L_BACKWARD_Pass_OUTPUTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_BACKWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Filter Backprop Outputs <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_Filter_BACKPROP_RESULTS[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Bias Outputs Before Update <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH; ++elementIndex)
		{
			cout << L_PrevBiases[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_BACKWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}


	cout << ">>>> Bias Outputs After Update <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH; ++elementIndex)
		{
			cout << L_Biases[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == L_FORWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Filter Adam Optimizer V Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_AdamOptimizer_V_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Filter Adam Optimizer >CORRECTED< V Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_AdamOptimizer_Corrected_V_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Filter Adam Optimizer S Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_AdamOptimizer_S_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Filter Adam Optimizer >CORRECTED< S Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_AdamOptimizer_Corrected_S_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> BIAS Adam Optimizer V Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_BIAS_AdamOptimizer_V_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> BIAS Adam Optimizer >CORRECTED< V Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_BIAS_AdamOptimizer_Corrected_V_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> BIAS Adam Optimizer S Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_BIAS_AdamOptimizer_S_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> BIAS Adam Optimizer >CORRECTED< S Matrix <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < L_NumberOf_FILTERS; ++inputIndex)
	{
		cout << "- Element " << inputIndex + 1 << "-" << endl;
		for (int elementIndex = 0; elementIndex < m_FilterSize * m_FilterSize; ++elementIndex)
		{
			cout << L_BIAS_AdamOptimizer_Corrected_S_Matrix[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == m_FilterSize + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

}
