#include "ErrorCalcModule.cuh"

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

__global__ void PixelWiseSigmoidKernel(float* d_Input, float* d_Output, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;
	float pixel = d_Input[arrayIndex];

	float SigmoidResult = 1 / 1 + exp(pixel);

	d_Output[arrayIndex] = SigmoidResult;
};

__global__ void PixelWiseCrossEntropyKernel(float* d_Input, float* d_Output, float* d_GroundTruthMatrix, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;

	float predictedPixel = d_Input[arrayIndex];
	float groundTruthClass = d_Input[arrayIndex];

	float correctedPixel = predictedPixel;
	if (groundTruthClass == 0)
	{
		correctedPixel = 1 - predictedPixel;
	}


	d_Output[arrayIndex] = -log(correctedPixel);
};

__global__ void LevelCrossEntropySumKernel(float* d_Input, float* d_Output, int matrixWidth)
{
	int blockStart = (blockIdx.x + 1) * 2;
	int rowNumber = blockStart + threadIdx.x;
	int arrayIndex = rowNumber * matrixWidth;

	float sum = 0;

	for (int i = 0; i < matrixWidth; ++i)
	{
		sum += d_Input[arrayIndex];
		arrayIndex++;
	}

	d_Output[arrayIndex] = sum;
};
__global__ void GradientGenerationKernel(float* d_Input, float* d_Output) {};


ErrorCalcModule::ErrorCalcModule(float* _inputMatrix, float* _groundTruth, int _inputHeight, int _inputWidth)
{
	m_InputMatrix = _inputMatrix;
	groundTruthMatrix = _groundTruth;
	m_InputMatrixHeight = _inputHeight;
	m_InputMatrixWidth = _inputWidth;

	m_OutputMatrixHeight = _inputHeight;
	m_OutputMatrixWidth = _inputWidth;

	m_OutputMatrix = new float[m_OutputMatrixHeight * m_OutputMatrixWidth];
}

void ErrorCalcModule::ForwardPass()
{
	PixelWiseSoftMax();
	PixelWiseCrossEntropy();
	CrossEntropySum();
}

void ErrorCalcModule::PixelWiseSoftMax()
{
	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);
	std::cout << "Pixel count " << totalPixelCount;

	sigmoidResultMatrix = new float[totalPixelCount];

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

	PixelWiseSigmoidKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(sigmoidResultMatrix, d_Output, byteCount, cudaMemcpyDeviceToHost);
}

void ErrorCalcModule::PixelWiseCrossEntropy()
{
	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);
	std::cout << "Pixel count " << totalPixelCount;

	crossEntropyResultMatrix = new float[totalPixelCount];

	//Define pointers for deviceMemory locations
	float* d_SigmoidInput;
	float* d_Output;
	float* d_GroundTruthMatrix;


	//Allocate memory
	cudaMalloc((void**)&d_SigmoidInput, byteCount);
	cudaMalloc((void**)&d_GroundTruthMatrix, byteCount);
	cudaMalloc((void**)&d_Output, byteCount);


	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_SigmoidInput, sigmoidResultMatrix, byteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_GroundTruthMatrix, groundTruthMatrix, byteCount, cudaMemcpyHostToDevice);


	//Define block size and threads per block.
	dim3 blockGrid(m_InputMatrixHeight, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth, 1, 1);

	PixelWiseCrossEntropyKernel << <blockGrid, threadGrid >> > (d_SigmoidInput, d_Output, d_GroundTruthMatrix, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(crossEntropyResultMatrix, d_Output, byteCount, cudaMemcpyDeviceToHost);
}


void ErrorCalcModule::CrossEntropySum()
{
	int blockCount = m_InputMatrixHeight / 2;
	
	float* intermediateSum = new float[m_InputMatrixHeight];


	size_t inputPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int inputByteCount = inputPixelCount * sizeof(float);
	int outputByteCount = m_InputMatrixHeight * sizeof(float);
	std::cout << "Pixel count " << inputPixelCount ;

	float* d_Input;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMemcpy(d_Input, crossEntropyResultMatrix, inputByteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(blockCount, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth, 1, 1);

	LevelCrossEntropySumKernel<< <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(intermediateSum, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

	networkError = 0;

	for (int i = 0; i < m_InputMatrixHeight; ++i)
	{
		networkError += intermediateSum[i];
	}
	cout << "Network error " << networkError << endl;
}




void ErrorCalcModule::BackwardPass()
{

}

void ErrorCalcModule::UpdateModule()
{

}


