#pragma once

#ifndef GRADIENT_GPU_H
#define GRADIENT_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class ErrorCalcModule : public CunetaModule
{
public:
	ErrorCalcModule(float* _inputMatrix, float*_groundTruth,int _inputHeight, int _inputWidth);
	float networkError;
	float* groundTruthMatrix;
	float* sigmoidResultMatrix;
	float* crossEntropyResultMatrix;
	float* intermediateSumResult;
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	void PixelWiseSigmoid();
	void PixelWiseCrossEntropy();
	void CrossEntropySum();
	
};

#endif
