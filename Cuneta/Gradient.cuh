#pragma once

#ifndef GRADIENT_GPU_H
#define GRADIENT_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Gradient : public CunetaModule
{
	Gradient(float* _inputMatrix, float* _outputMatrix);
	float* filter;
	int fliterSize;
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	
	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
	__global__ void PizelWiseSoftMaxKernel(float* d_Input, float* d_Output) {};
	__global__ void PixelWiseCrossEntropyKernel(float* d_Input, float* d_Output) {};
	__global__ void CrossEntropySumKernel(float* d_Input, float* d_Output) {};
	__global__ void GradientGenerationKernel(float* d_Input, float* d_Output) {};

};

#endif
