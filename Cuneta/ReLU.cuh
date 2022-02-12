#pragma once

#ifndef RELU_GPU_H
#define RELU_GPU_H

#include "cuda_runtime.h"
#include "CunetaModule.cuh"


class ReLU : public CunetaModule
{
public: 
	ReLU(int _numberOfInputs, int _numberOfOutputs, int _IOHeight, int _IOWidth, int _layerID, int _levelID);
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;

	void SingleChannelForwardPass(float* input, float* output);
	void LayerForwardPass(float** _inputs) override;
	void LayerBackwardPass(float** _backpropInput) override;
	void UpdateModule() override;
	void PrintLayerParams();
};

#endif
