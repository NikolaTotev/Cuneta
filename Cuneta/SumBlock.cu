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
	int inputRowIndex = blockIdx.x * 2;
	int inputColumnIndex = threadIdx.x * 2;


	int inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;


	for (int row = 0; row < 2; row++)
	{
		inputColumnIndex = threadIdx.x * 2;
		inputRowIndex += row;


		for (int col = 0; col < 2; col++)
		{
			inputColumnIndex += col;

			inputArrayIndex = inputRowIndex * _inputWidth + inputColumnIndex;

			_output[inputArrayIndex] = _input_1[inputArrayIndex] + _input_2[inputArrayIndex];
		}
	}
}

SumBlock::SumBlock(int _height, int _width, int _numberOfElements)
{
	Height = _height;
	Width = _width;
	NumberOfElements = _numberOfElements;

	Output = new float* [NumberOfElements];
	InputSet_1 = new float* [NumberOfElements];
	InputSet_2 = new float* [NumberOfElements];

	for (int i = 0; i < NumberOfElements; ++i)
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

	for (int i = 0; i < NumberOfElements; ++i)
	{
		memcpy(InputSet_1[i], _inputSet_1[i], inputByteCount);
		memcpy(InputSet_2[i], _inputSet_2[i], inputByteCount);
	}

	for (int inputNumber = 0; inputNumber < NumberOfElements; ++inputNumber)
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


