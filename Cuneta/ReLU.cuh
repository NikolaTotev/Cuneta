#pragma once

#ifndef RELU_GPU_H
#define RELU_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class ReLU : public CunetaModule
{
	ReLU(float* _inputMatrix, float* _outputMatrix);
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;

	//Input and output will be in global memory.
	__global__ void ReLUKernel(float* d_Input, float* d_Output){};
};

#endif
