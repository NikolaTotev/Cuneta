#pragma once

#ifndef MAXPOOL_GPU_H
#define MAXPOOL_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class MaxPool : public CunetaModule
{
public:
	MaxPool(int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth, int _layerID, int _levelID);
	MaxPool();
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void LayerForwardPass(float** _inputs) override;
	void LayerBackwardPass(float** _backpropInput) override;
	void UpdateModule() override;
	void Print();
	void PrintLayerParams();
	void DebugPrintAll();

	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
};

#endif
