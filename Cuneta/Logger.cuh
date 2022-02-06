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


class CunetaLogger
{
public:
	void LogReLUState(ReLU reluToSave);
	void LogMaxPoolState(MaxPool maxPoolToSave);
	void LogConvolutionState(Convolution convolutionToSave);
	void LogTransposeConvolutionState(TransposeConvolution transpConvToSave);
	void LogErrorState(ErrorCalcModule errorModuleToSave);
	void AddErrorScore(float scoreToAdd);
};
