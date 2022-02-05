#pragma once

#ifndef RELU_GPU_H
#define RELU_GPU_H

#include "cuda_runtime.h"
#include "CunetaModule.cuh"


class ReLU : public CunetaModule
{
public: 
	ReLU();
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void UpdateModule() override;

};

#endif
