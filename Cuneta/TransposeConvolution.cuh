#pragma once

#ifndef TRANSPOSE_CONVOLUTION_GPU_H
#define TRANSPOSE_CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class TransposeConvolution : public CunetaModule
{
	TransposeConvolution(float* _inputMatrix, float* _outputMatrix);
	float* filter;
	int fliterSize;
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	void GenerateToeplitzMatrix();
	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
	__global__ void TransposeConvolutionKernel(float* d_Input, float* d_Output) {};

};

#endif
