
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

__global__ void MaxPoolKernel(float* d_Input, float* d_Output, int _inputWidth, int _outputWidth)
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


__global__ void BackpropMaxPoolKernel(float* d_BackpropInput, float* d_FwdInput, float* d_Output, int _fwdInputWidth, int _backInputWidth)
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
	float currentPixel = d_FwdInput[fwdInputArrayIndex]; ///OK
	int maxElementIndex = fwdInputArrayIndex; ///OK

	for (int row = 0; row < 2; row++)
	{
		fwdInputColumnIndex = threadIdx.x * 2; ///OK

		fwdInputRowIndex += row; ///OK


		for (int col = 0; col < 2; col++)
		{
			fwdInputColumnIndex += col;

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

MaxPool::MaxPool(int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth, int _layerID, int _levelID)
{
	L_FORWARD_NumberOf_INPUTS = _numberOfInputs;
	L_FORWARD_NumberOf_OUTPUTS = _numberOfOutputs;

	L_BACKWARD_NumberOf_INPUTS = L_FORWARD_NumberOf_OUTPUTS;
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
	L_FORWARD_Pass_OUTPUTS = new float* [L_FORWARD_NumberOf_OUTPUTS];

	L_BACKWARD_Pass_INPUTS = new float* [L_BACKWARD_NumberOf_INPUTS];
	L_BACKWARD_Pass_OUTPUTS = new float* [L_BACKWARD_NumberOf_OUTPUTS];

	levelID = _levelID;
	layerID = _layerID;
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

	MaxPoolKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth, m_OutputMatrixWidth);
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
	m_BackpropagationOutput = new float[m_BackpropOutputMatrixHeight * m_BackpropOutputMatrixWidth]; ///OK

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

	BackpropMaxPoolKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixWidth, m_BackpropInputMatrixWidth);
	cudaDeviceSynchronize(); ///OK

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, outputByteCount, cudaMemcpyDeviceToHost); ///OK
}


void MaxPool::LayerForwardPass(float** _inputs)
{
	for (int inputNumber = 0; inputNumber < L_FORWARD_NumberOf_INPUTS; ++inputNumber)
	{
		int inputSize = L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH;
		int outputSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;
		size_t inputByteCount = inputSize * sizeof(float);
		size_t outputByteCount = outputSize * sizeof(float);

		L_FORWARD_Pass_INPUTS[inputNumber] = new float[inputSize];
		L_FORWARD_Pass_OUTPUTS[inputNumber] = new float[outputSize];

		memcpy(L_FORWARD_Pass_INPUTS[inputNumber], _inputs[inputNumber], inputByteCount);

		//Define pointers for deviceMemory locations
		float* d_Input;
		float* d_Output;

		//Allocate memory
		cudaMalloc((void**)&d_Input, inputByteCount);
		cudaMalloc((void**)&d_Output, outputByteCount);

		//Copy memory into global device memory m_InputMatrix -> d_Input
		cudaMemcpy(d_Input, L_FORWARD_Pass_INPUTS[inputNumber], inputByteCount, cudaMemcpyHostToDevice);

		//Define block size and threads per block.
		dim3 blockGrid(L_FORWARD_InputLayer_HEIGHT / 2, 1, 1);
		dim3 threadGrid(L_FORWARD_InputLayer_WIDTH / 2, 1, 1);

		MaxPoolKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, L_FORWARD_InputLayer_WIDTH, L_FORWARD_OutputLayer_WIDTH);
		cudaDeviceSynchronize();

		//Copy back result into host memory d_Output -> m_OutputMatrix
		cudaMemcpy(L_FORWARD_Pass_OUTPUTS[inputNumber], d_Output, outputByteCount, cudaMemcpyDeviceToHost);
		cudaFree(d_Input);
		cudaFree(d_Output);
	}
}


void MaxPool::LayerBackwardPass(float** _backpropInput)
{
	int forwardInputSize = L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH;

	int backwardInputSize = L_BACKWARD_InputLayer_HEIGHT * L_BACKWARD_InputLayer_WIDTH;

	int backwardOutputSize = L_BACKWARD_OutputLayer_HEIGHT * L_BACKWARD_OutputLayer_WIDTH;

	size_t forwardInputByteCount = forwardInputSize * sizeof(float);

	size_t backwardInputByteCount = backwardInputSize * sizeof(float);

	size_t backwardOutputByteCount = backwardOutputSize * sizeof(float);

	for (int inputNumber = 0; inputNumber < L_BACKWARD_NumberOf_INPUTS; ++inputNumber)
	{

		L_BACKWARD_Pass_INPUTS[inputNumber] = new float[backwardInputSize];
		L_BACKWARD_Pass_OUTPUTS[inputNumber] = new float[backwardOutputSize];

		memcpy(L_BACKWARD_Pass_INPUTS[inputNumber], _backpropInput[inputNumber], backwardInputByteCount);

		//Define pointers for deviceMemory locations
		float* d_FwdInput; ///OK
		float* d_BackpropInput; ///OK
		float* d_BackwardOutput; ///OK

		//Allocate memory
		cudaMalloc((void**)&d_FwdInput, forwardInputByteCount); ///OK
		cudaMalloc((void**)&d_BackpropInput, backwardInputByteCount);///OK
		cudaMalloc((void**)&d_BackwardOutput, backwardOutputByteCount);///OK

		//Copy memory into global device memory m_InputMatrix -> d_Input
		cudaMemcpy(d_FwdInput, L_FORWARD_Pass_INPUTS[inputNumber], forwardInputByteCount, cudaMemcpyHostToDevice);
		cudaMemcpy(d_BackpropInput, L_BACKWARD_Pass_INPUTS[inputNumber], backwardInputByteCount, cudaMemcpyHostToDevice);
		///OK
		///OK

		//Define block size and threads per block.
		dim3 blockGrid(L_BACKWARD_InputLayer_HEIGHT, 1, 1); ///OK
		dim3 threadGrid(L_BACKWARD_InputLayer_WIDTH, 1, 1);///OK

		BackpropMaxPoolKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_BackwardOutput, L_FORWARD_InputLayer_WIDTH, L_BACKWARD_InputLayer_WIDTH); ///OK
		cudaDeviceSynchronize();///OK

		//Copy back result into host memory d_Output -> m_OutputMatrix
		cudaMemcpy(L_BACKWARD_Pass_OUTPUTS[inputNumber], d_BackwardOutput, backwardOutputByteCount, cudaMemcpyDeviceToHost);///OK

		cudaFree(d_FwdInput);
		cudaFree(d_BackpropInput);
		cudaFree(d_BackwardOutput);
	}
}

void MaxPool::PrintLayerParams()
{
	cout << "===================================" << endl;
	cout << "====== ReLU Layer Parameters ======" << endl;
	cout << "===================================" << endl;
	cout << "ReLU: Layer " << layerID << " " << "Level " << levelID << endl;

	cout << endl;

	cout << "-- Forward Dimensions --" << endl;
	cout << "Forward Input Height: " << L_FORWARD_InputLayer_HEIGHT << " || Forward Output Height: " << L_FORWARD_OutputLayer_HEIGHT << endl;
	cout << "Forward Input Width: " << L_FORWARD_InputLayer_WIDTH << " || Forward Output Width: " << L_FORWARD_OutputLayer_WIDTH << endl;

	cout << endl;

	cout << "-- Backward Dimensions --" << endl;
	cout << "Backward Input Height: " << L_BACKWARD_InputLayer_HEIGHT << " || Forward Output Height: " << L_BACKWARD_OutputLayer_HEIGHT << endl;
	cout << "Backward Input Width: " << L_BACKWARD_InputLayer_WIDTH << " || Forward Output Width: " << L_BACKWARD_OutputLayer_WIDTH << endl;

	cout << endl;

	cout << "-- Feature map count --" << endl;
	cout << "Forward Input Count: " << L_FORWARD_NumberOf_INPUTS << " || Backward Input Count: " << L_BACKWARD_NumberOf_INPUTS << endl;
	cout << "Forward Output Count: " << L_FORWARD_NumberOf_OUTPUTS << " || Backward Output Count: " << L_BACKWARD_NumberOf_OUTPUTS << endl;

	cout << "===================================" << endl;
}

void MaxPool::UpdateModule()
{

}

void MaxPool::DebugPrintAll()
{
	int newLineCounter = 1;

	cout << "=================================================" << endl;
	cout << "============ MaxPool Debug Print All ============" << endl;
	cout << "=================================================" << endl;

	cout << "Squishy: " << endl;
	cout << "Layer ID: " << layerID << endl;
	cout << "Level ID: " << levelID << endl;


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
}






