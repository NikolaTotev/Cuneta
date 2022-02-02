#pragma once

#ifndef MAXPOOL_GPU_H
#define MAXPOOL_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class MaxPool : public CunetaModule
{
	MaxPool(float* _inputMatrix, float* _outputMatrix);
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;

	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
	__global__ void MaxPoolKernel(float* d_Input, float* d_Output) {};
};

#endif
