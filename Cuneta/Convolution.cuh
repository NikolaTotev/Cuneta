#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
	Convolution(float* _inputMatrix, float* _outputMatrix);
	float* filter;
	int fliterSize;
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	void Dialate(float* _input, float *_output);

	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
	__global__ void ConvolutionKernel(float* d_Input, float* d_Output) {};
};

#endif
