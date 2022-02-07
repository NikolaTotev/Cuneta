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


class CunetaLogger
{
public:
	void LogReLUState(ReLU reluToSave, string outputDirectory, string imageName, int iteration);
	void LogMaxPoolState(MaxPool maxPoolToSave, string outputDirectory, string imageName, int iteration);
	void LogConvolutionState(Convolution convolutionToSave, string outputDirectory, string imageName, int iteration);
	void LogTransposeConvolutionState(TransposeConvolution transposeConvolutionToSave, string outputDirectory, string imageName, int iteration);
	void LogErrorState(ErrorCalcModule errorModuleToSave, string outputDirectory, string imageName, int iteration);
	void AddErrorScore(float scoreToAdd);
	void AddImageNameToProcessingHistory(string outputDirectory, string imagePath);
};
