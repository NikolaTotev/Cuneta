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
	float* dLdXMatrix;
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void UpdateModule() override;
	void PixelWiseSigmoid();
	void PixelWiseCrossEntropy();
	void CrossEntropySum();
	void CalculateGradient();
	
};

#endif
