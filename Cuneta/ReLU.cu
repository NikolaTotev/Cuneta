#include "ReLU.cuh"

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

//Input and output will be in global memory. d_ shows in which memory the variables are stored.
__global__ void ReLUKernel(float* d_Input, float* d_Output, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;
	float pixel = d_Input[arrayIndex];

	float ReLUResult = fmaxf(0, pixel);

	d_Output[arrayIndex] = ReLUResult;
}


__global__ void BackpropReLUKernel(float* d_BackpropInput, float* d_FwdInput, float* d_BackpropOutput, int matrixWidth)
{
	int rowIndex = blockIdx.x; ///OK
	int columnIndex = threadIdx.x;///OK
	int arrayIndex = rowIndex * matrixWidth + columnIndex; ///OK
	float fwdInputPixel = d_FwdInput[arrayIndex];///OK
	float backpropInputPixel = d_BackpropInput[arrayIndex];///OK

	float ReLUResult = 0; ///OK
	if (fwdInputPixel > 0)
	{
		ReLUResult = 1 * backpropInputPixel;
	}

	d_BackpropOutput[arrayIndex] = ReLUResult;///OK	

}

ReLU::ReLU(int _numberOfInputs, int _numberOfOutputs, int _IOHeight, int _IOWidth, int _layerID, int _levelID)
{
	L_FORWARD_NumberOf_INPUTS = _numberOfInputs;
	L_FORWARD_NumberOf_OUTPUTS = _numberOfOutputs;

	L_BACKWARD_NumberOf_INPUTS = L_FORWARD_NumberOf_OUTPUTS;
	L_BACKWARD_NumberOf_OUTPUTS = L_FORWARD_NumberOf_INPUTS;

	L_FORWARD_InputLayer_HEIGHT = _IOHeight;
	L_FORWARD_InputLayer_WIDTH = _IOWidth;

	L_FORWARD_OutputLayer_HEIGHT = _IOHeight;
	L_FORWARD_OutputLayer_WIDTH = _IOWidth;

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


void ReLU::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{
	int arrayLength = fwdPassHeight * fwdPassWidth;
	size_t inputSize = arrayLength * sizeof(float);

	m_InputMatrixHeight = fwdPassHeight;
	m_InputMatrixWidth = fwdPassWidth;

	m_OutputMatrixHeight = m_InputMatrixHeight;
	m_OutputMatrixWidth = m_InputMatrixWidth;

	m_InputMatrix = new float[arrayLength];
	m_OutputMatrix = new float[arrayLength];

	memcpy(m_InputMatrix, forwardPassInput, inputSize);

	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);
	std::cout << "Pixel count " << totalPixelCount;

	//Define pointers for deviceMemory locations
	float* d_Input;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_Input, byteCount);
	cudaMalloc((void**)&d_Output, byteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_Input, m_InputMatrix, byteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_InputMatrixHeight, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth, 1, 1);

	ReLUKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_OutputMatrix, d_Output, byteCount, cudaMemcpyDeviceToHost);

}

void ReLU::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{

	int arrayLength = backPassHeight * backPassWidth; ///OK
	size_t inputSize = arrayLength * sizeof(float); ///OK

	m_BackpropInputMatrixHeight = backPassHeight; ///OK
	m_BackpropInputMatrixWidth = backPassWidth; ///OK

	m_BackpropOutputMatrixHeight = m_BackpropInputMatrixHeight;///OK
	m_BackpropOutputMatrixWidth = m_BackpropInputMatrixWidth;///OK

	m_BackPropInputMatrix = new float[arrayLength];///OK
	m_BackpropagationOutput = new float[arrayLength];///OK

	memcpy(m_BackPropInputMatrix, backpropInput, inputSize); ///OK

	size_t totalPixelCount = m_BackpropInputMatrixHeight * m_BackpropInputMatrixWidth; ///OK
	int byteCount = totalPixelCount * sizeof(float); ///OK

	//Define pointers for deviceMemory locations
	float* d_FwdInput; ///OK
	float* d_BackpropInput; ///OK
	float* d_Output; ///OK

	//Allocate memory
	cudaMalloc((void**)&d_FwdInput, byteCount); ///OK
	cudaMalloc((void**)&d_BackpropInput, byteCount);///OK
	cudaMalloc((void**)&d_Output, byteCount);///OK

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_BackpropInput, m_BackPropInputMatrix, byteCount, cudaMemcpyHostToDevice); ///OK
	cudaMemcpy(d_FwdInput, m_InputMatrix, byteCount, cudaMemcpyHostToDevice); ///OK

	//Define block size and threads per block.
	dim3 blockGrid(m_BackpropInputMatrixHeight, 1, 1); ///OK
	dim3 threadGrid(m_BackpropInputMatrixWidth, 1, 1);///OK

	BackpropReLUKernel << <blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_Output, m_InputMatrixWidth); ///OK
	cudaDeviceSynchronize();///OK

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(m_BackpropagationOutput, d_Output, byteCount, cudaMemcpyDeviceToHost);///OK
}


void ReLU::LayerForwardPass(float** _inputs)
{
	int inputSize = L_FORWARD_InputLayer_HEIGHT * L_FORWARD_InputLayer_WIDTH;
	int outputSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;
	size_t inputByteCount = inputSize * sizeof(float);
	size_t outputByteCount = inputSize * sizeof(float);

	for (int inputNumber = 0; inputNumber < L_FORWARD_NumberOf_INPUTS; ++inputNumber)
	{
		L_FORWARD_Pass_INPUTS[inputNumber] = new float[inputSize];
		L_FORWARD_Pass_OUTPUTS[inputNumber] = new float[outputSize];

		memcpy(L_FORWARD_Pass_INPUTS[inputNumber], _inputs[inputNumber], inputByteCount);

		//Define pointers for deviceMemory locations
		float* d_Input;
		float* d_Output;

		//Allocate memory
		cudaMalloc((void**)&d_Input, inputByteCount);
		cudaMalloc((void**)&d_Output, inputByteCount);

		//Copy memory into global device memory m_InputMatrix -> d_Input
		cudaMemcpy(d_Input, L_FORWARD_Pass_INPUTS[inputNumber], inputByteCount, cudaMemcpyHostToDevice);

		//Define block size and threads per block.
		dim3 blockGrid(L_FORWARD_InputLayer_HEIGHT, 1, 1);
		dim3 threadGrid(L_FORWARD_InputLayer_WIDTH, 1, 1);

		ReLUKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, L_FORWARD_InputLayer_WIDTH);
		cudaDeviceSynchronize();

		//Copy back result into host memory d_Output -> m_OutputMatrix
		cudaMemcpy(L_FORWARD_Pass_OUTPUTS[inputNumber], d_Output, outputByteCount, cudaMemcpyDeviceToHost);
	}
}


void ReLU::LayerBackwardPass(float** _backpropInput)
{
	for (int inputNumber = 0; inputNumber < L_BACKWARD_NumberOf_INPUTS; ++inputNumber)
	{
		int forwardInputSize = L_FORWARD_OutputLayer_HEIGHT * L_FORWARD_OutputLayer_WIDTH;

		int backwardInputSize = L_BACKWARD_InputLayer_HEIGHT * L_BACKWARD_InputLayer_WIDTH;

		int backwardOutputSize = L_BACKWARD_OutputLayer_HEIGHT * L_BACKWARD_OutputLayer_WIDTH;

		size_t forwardInputByteCount = forwardInputSize * sizeof(float);

		size_t backwardInputByteCount = backwardInputSize * sizeof(float);

		size_t backwardOutputByteCount = backwardOutputSize * sizeof(float);


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
		cudaMemcpy(d_BackpropInput, L_BACKWARD_Pass_INPUTS[inputNumber], backwardInputByteCount, cudaMemcpyHostToDevice); ///OK
		cudaMemcpy(d_FwdInput, L_FORWARD_Pass_INPUTS[inputNumber], forwardInputByteCount, cudaMemcpyHostToDevice); ///OK

		//Define block size and threads per block.
		dim3 blockGrid(L_FORWARD_InputLayer_HEIGHT, 1, 1);
		dim3 threadGrid(L_FORWARD_InputLayer_WIDTH, 1, 1);

		BackpropReLUKernel << < blockGrid, threadGrid >> > (d_BackpropInput, d_FwdInput, d_BackwardOutput, L_FORWARD_OutputLayer_WIDTH); ///OK
		cudaDeviceSynchronize();///OK
		float* temp = new float[backwardOutputSize];

		//Copy back result into host memory d_Output -> m_OutputMatrix
		cudaMemcpy(temp, d_BackwardOutput, backwardOutputByteCount, cudaMemcpyDeviceToHost);///OK
		memcpy(L_BACKWARD_Pass_OUTPUTS[inputNumber], temp, backwardOutputByteCount);

	}
}

void ReLU::PrintLayerParams()
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


void ReLU::UpdateModule()
{

}


void ReLU::DebugPrintAll()
{
	int newLineCounter = 1;

	cout << "=================================================" << endl;
	cout << "============ ReLU Debug Print All ============" << endl;
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
	newLineCounter = 1;


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
	newLineCounter = 1;


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
	newLineCounter = 1;


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





