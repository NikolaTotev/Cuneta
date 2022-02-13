#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include <stdio.h>

#include "Convolution.cuh"
#include "ErrorCalcModule.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "TransposeConvolution.cuh"
using namespace std;

#ifndef LOGGER_GPU_H
#define LOGGER_GPU_H
class CunetaLogger
{
public:
	void LogReLUState(ReLU reluToSave, string outputDirectory, string imageName, int iteration);
	void LogMaxPoolState(MaxPool maxPoolToSave, string outputDirectory, string imageName, int iteration);
	void LogConvolutionState(Convolution convolutionToSave, string outputDirectory, string imageName, int iteration);
	void LogTransposeConvolutionState(TransposeConvolution transposeConvolutionToSave, string outputDirectory, string imageName, int iteration);
	void LogErrorState(ErrorCalcModule errorModuleToSave, string outputDirectory, string imageName, int iteration);
	void AddErrorScore(string outputDirectory, float scoreToAdd, int iteration);
	void AddImageNameToProcessingHistory(string outputDirectory, string imagePath, int iteration);
	void SaveFilter(float* filter, int filterSize, string outputDirectory, string layer, int iteration);
	void SaveOutput(float* cunetaOutput, int height, int width, string outputDirectory, string layer, int iteration, int ephoc);
	void Save_RELU_Test(ReLU testSubject, string outputDirectory, int testNumber);
	void Save_MAXPOOL_Test(ReLU reluToSave, string outputDirectory, int testNumber);
	void Save_CONV_Test(ReLU reluToSave, string outputDirectory, int testNumber);
	void Save_TCONV_Test(ReLU reluToSave, string outputDirectory, int testNumber);
	void Save_SQUISH_Test(ReLU reluToSave, string outputDirectory, int testNumber);
	void Save_SUMBLOCK_Test(ReLU reluToSave, string outputDirectory, int testNumber);
	void Save_ERR_Test(ReLU reluToSave, string outputDirectory, int testNumber);
};

#endif

