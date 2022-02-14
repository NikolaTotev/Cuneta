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

	float SigmoidResult = 1 / (1 + exp(-pixel));

	d_Output[arrayIndex] = SigmoidResult;
};

__global__ void GradientKernel(float* d_PredictedProbabilityMatrix, float* dCrossEntropy_dRawInput, float* groundTruth, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;
	float predictedProbability = d_PredictedProbabilityMatrix[arrayIndex];

	float dCrossEntropy_dPredictedProbability = -1 / predictedProbability;

	/*if(groundTruth[arrayIndex] == 0)
	{
		dCrossEntropy_dPredictedProbability = predictedProbability-1;

	}
	else{
		dCrossEntropy_dPredictedProbability = predictedProbability;
	}*/

	float dPredictedProbablity_dRawInput = predictedProbability * (1 - predictedProbability);

	dCrossEntropy_dRawInput[arrayIndex] = dCrossEntropy_dPredictedProbability * dPredictedProbablity_dRawInput;
}

__global__ void PixelWiseCrossEntropyKernel(float* d_Input, float* d_Output, float* d_GroundTruthMatrix, int matrixWidth)
{
	int rowIndex = blockIdx.x;
	int columnIndex = threadIdx.x;
	int arrayIndex = rowIndex * matrixWidth + columnIndex;

	float predictedPixel = d_Input[arrayIndex];
	float groundTruthClass = d_GroundTruthMatrix[arrayIndex];

	float correctedPixel = predictedPixel;
	if (groundTruthClass == 0)
	{
		correctedPixel = 1 - predictedPixel;
	}


	d_Output[arrayIndex] = -log(correctedPixel);
};

__global__ void CrossEntropySumKernel(float* d_Input, float* d_Output, int matrixWidth, int matrixHeight)
{
	int blockStart = blockIdx.x * 2;
	int rowNumber = blockStart + threadIdx.x;
	int arrayIndex = rowNumber * matrixWidth;

	float sum = 0;

	for (int i = 0; i < matrixWidth; ++i)
	{
		sum += d_Input[arrayIndex];
		arrayIndex++;
	}
	if (blockIdx.x == 0 && threadIdx.x == 1)
	{
		//	d_Output[0] = arrayIndex;
	}
	d_Output[rowNumber] = sum;


};


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

void ErrorCalcModule::ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth)
{

}

void ErrorCalcModule::LayerForwardPass()
{
	PixelWiseSigmoid();
	PixelWiseCrossEntropy();
	CrossEntropySum();
	CalculateGradient();
}


void ErrorCalcModule::PixelWiseSigmoid()
{
	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);

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

	intermediateSumResult = new float[m_InputMatrixHeight];


	size_t inputPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int inputByteCount = inputPixelCount * sizeof(float);
	int outputByteCount = m_InputMatrixHeight * sizeof(float);

	float* d_Input;
	float* d_Output;

	//Allocate memory
	cudaMalloc((void**)&d_Input, inputByteCount);
	cudaMalloc((void**)&d_Output, outputByteCount);
	cudaMemcpy(d_Input, crossEntropyResultMatrix, inputByteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(blockCount, 1, 1);
	dim3 threadGrid(2, 1, 1);

	CrossEntropySumKernel << <blockGrid, threadGrid >> > (d_Input, d_Output, m_InputMatrixWidth, m_InputMatrixHeight);
	cudaDeviceSynchronize();

	cudaMemcpy(intermediateSumResult, d_Output, outputByteCount, cudaMemcpyDeviceToHost);

	networkError = 0;

	for (int i = 0; i < m_InputMatrixHeight; ++i)
	{
		networkError += intermediateSumResult[i];
		cout << intermediateSumResult[i] << " ";
	}
}

void ErrorCalcModule::CalculateGradient()
{

	size_t totalPixelCount = m_InputMatrixHeight * m_InputMatrixWidth;
	int byteCount = totalPixelCount * sizeof(float);

	dLdXMatrix = new float[totalPixelCount];

	//Define pointers for deviceMemory locations
	float* d_predictedProb;
	float* d_dLdX;
	float* d_GroundTruth;

	//Allocate memory
	cudaMalloc((void**)&d_predictedProb, byteCount);
	cudaMalloc((void**)&d_dLdX, byteCount);
	cudaMalloc((void**)&d_GroundTruth, byteCount);

	//Copy memory into global device memory m_InputMatrix -> d_Input
	cudaMemcpy(d_predictedProb, sigmoidResultMatrix, byteCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_GroundTruth, groundTruthMatrix, byteCount, cudaMemcpyHostToDevice);

	//Define block size and threads per block.
	dim3 blockGrid(m_InputMatrixHeight, 1, 1);
	dim3 threadGrid(m_InputMatrixWidth, 1, 1);

	GradientKernel << <blockGrid, threadGrid >> > (d_predictedProb, d_dLdX, groundTruthMatrix, m_InputMatrixWidth);
	cudaDeviceSynchronize();

	//Copy back result into host memory d_Output -> m_OutputMatrix
	cudaMemcpy(dLdXMatrix, d_dLdX, byteCount, cudaMemcpyDeviceToHost);
}



void ErrorCalcModule::BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth)
{

}

void ErrorCalcModule::UpdateModule()
{

}

void ErrorCalcModule::DebugPrintAll()
{
	int newLineCounter = 1;

	cout << "===========================================================" << endl;
	cout << "============ Error Calc Module Debug Print All ============" << endl;
	cout << "===========================================================" << endl;

	cout << "Error Calc: " << endl;

	cout << ">>>> Input <<<<" << endl << endl;
	newLineCounter = 1;

	for (int i = 0; i < m_InputMatrixHeight * m_InputMatrixWidth; ++i)
	{
		cout << m_InputMatrix[i] << " ";
		newLineCounter++;
		if (newLineCounter == m_InputMatrixWidth + 1)
		{
			cout << endl;
			newLineCounter = 1;
		}
	}

	cout << endl;
	cout << ">>>> Ground truth <<<<" << endl << endl;
	newLineCounter = 1;

	for (int i = 0; i < m_InputMatrixHeight * m_InputMatrixWidth; ++i)
	{
		cout << groundTruthMatrix[i] << " ";
		newLineCounter++;
		if (newLineCounter == m_InputMatrixWidth + 1)
		{
			cout << endl;
			newLineCounter = 1;
		}
	}

	cout << endl;
	cout << ">>>> Sigmoid Result <<<<" << endl << endl;
	newLineCounter = 1;

	for (int i = 0; i < m_InputMatrixHeight * m_InputMatrixWidth; ++i)
	{
		cout << sigmoidResultMatrix[i] << " ";
		newLineCounter++;
		if (newLineCounter == m_InputMatrixWidth + 1)
		{
			cout << endl;
			newLineCounter = 1;
		}
	}

	cout << endl;
	cout << ">>>> Cross Entropy Result <<<<" << endl << endl;
	newLineCounter = 1;

	for (int i = 0; i < m_InputMatrixHeight * m_InputMatrixWidth; ++i)
	{
		cout << crossEntropyResultMatrix[i] << " ";
		newLineCounter++;
		if (newLineCounter == m_InputMatrixWidth + 1)
		{
			cout << endl;
			newLineCounter = 1;
		}
	}

	cout << endl;
	cout << ">>>> Intermediate Sum Result <<<<" << endl << endl;

	for (int i = 0; i < m_InputMatrixHeight; ++i)
	{
		cout << intermediateSumResult[i] << endl;
	}

	cout << endl;
	cout << "Total Sum (Network error): " << networkError << endl;

	cout << endl;
	cout << ">>>> dXdL Result <<<<" << endl << endl;
	newLineCounter = 1;

	for (int i = 0; i < m_InputMatrixHeight * m_InputMatrixWidth; ++i)
	{
		cout << dLdXMatrix[i] << " ";
		newLineCounter++;
		if (newLineCounter == m_InputMatrixWidth + 1)
		{
			cout << endl;
			newLineCounter = 1;
		}
	}
	cout << endl;
	cout << endl;
}



