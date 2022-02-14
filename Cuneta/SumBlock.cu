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
#include "SumBlock.cuh"

__global__ void SumKernel(float* _input_1, float* _input_2, float* _output, int _inputWidth)
{
	//Starts from "top left" of current block of pixels being processed
	int inputRowIndex = blockIdx.x;
	int inputColumnIndex = 0;


	int inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;
	
		for (int col = 0; col < _inputWidth; col++)
		{
			inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;

			_output[inputArrayIndex] = _input_1[inputArrayIndex] + _input_2[inputArrayIndex];
			inputColumnIndex ++;
		}
}

SumBlock::SumBlock(int _height, int _width, int _numberOfLayers, int _layerID, int _levelID)
{
	Height = _height;
	Width = _width;
	NumberOfLayers = _numberOfLayers;

	levelID = _levelID;
	layerID = _layerID;

	Output = new float* [NumberOfLayers];
	InputSet_1 = new float* [NumberOfLayers];
	InputSet_2 = new float* [NumberOfLayers];

	for (int i = 0; i < NumberOfLayers; ++i)
	{
		Output[i] = new float[Height * Width];
		InputSet_1[i] = new float[Height * Width];
		InputSet_2[i] = new float[Height * Width];
	}
}



void SumBlock::Sum(float** _inputSet_1, float** _inputSet_2)
{
	int inputSize = Height * Width;

	size_t inputByteCount = inputSize * sizeof(float);

	for (int i = 0; i < NumberOfLayers; ++i)
	{
		memcpy(InputSet_1[i], _inputSet_1[i], inputByteCount);
		memcpy(InputSet_2[i], _inputSet_2[i], inputByteCount);
	}

	for (int inputNumber = 0; inputNumber < NumberOfLayers; ++inputNumber)
	{
		//Define pointers for deviceMemory locations
		float* d_Input_1;
		float* d_Input_2;
		float* d_Output;

		//Allocate memory
		cudaMalloc((void**)&d_Input_1, inputByteCount);
		cudaMalloc((void**)&d_Input_2, inputByteCount);
		cudaMalloc((void**)&d_Output, inputByteCount);

		//Copy memory into global device memory m_InputMatrix -> d_Input
		cudaMemcpy(d_Input_1, InputSet_1[inputNumber], inputByteCount, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Input_2, InputSet_2[inputNumber], inputByteCount, cudaMemcpyHostToDevice);

		//Define block size and threads per block.
		dim3 blockGrid(Height, 1, 1);
		dim3 threadGrid(Width, 1, 1);

		SumKernel << <blockGrid, threadGrid >> > (d_Input_1, d_Input_2, d_Output, Width);
		cudaDeviceSynchronize();

		//Copy back result into host memory d_Output -> m_OutputMatrix
		cudaMemcpy(Output[inputNumber], d_Output, inputByteCount, cudaMemcpyDeviceToHost);
	}
}

void SumBlock::DebugPrintAll()
{
	int newLineCounter = 1;

	cout << "===================================================" << endl;
	cout << "============ Sum Block Debug Print All ============" << endl;
	cout << "===================================================" << endl;

	cout << "Sum Block: " << endl;
	cout << "Layer ID: " << layerID << endl;
	cout << "Level ID: " << levelID << endl;
	cout << "Number of layers: "<< NumberOfLayers << endl;
	

	cout << ">>>> Input Set 1 <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < NumberOfLayers; ++inputIndex)
	{
		cout << "--- Element " << inputIndex + 1 << "---" << endl;
		for (int elementIndex = 0; elementIndex < Height*Width; ++elementIndex)
		{
			cout << InputSet_1[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == Width + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Input Set 2 <<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < NumberOfLayers; ++inputIndex)
	{
		cout << "--- Element " << inputIndex + 1 << "---" << endl;
		for (int elementIndex = 0; elementIndex < Height * Width; ++elementIndex)
		{
			cout << InputSet_2[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == Width + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

	cout << ">>>> Output Set<<<<" << endl << endl;

	for (int inputIndex = 0; inputIndex < NumberOfLayers; ++inputIndex)
	{
		cout << "--- Element " << inputIndex + 1 << "---" << endl;
		for (int elementIndex = 0; elementIndex < Height * Width; ++elementIndex)
		{
			cout << Output[inputIndex][elementIndex] << " ";
			newLineCounter++;
			if (newLineCounter == Width + 1)
			{
				cout << endl;
				newLineCounter = 1;
			}
		}
		cout << endl;
	}

}





