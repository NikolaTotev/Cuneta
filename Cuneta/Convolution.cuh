#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
public:
	Convolution(float* _inputMatrix, int _inputHeight, int _inputWidth, int _filterSize);
	float* filter;
	int filterSize;
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	void Dialate(float* _input, float *_output);
	void InitializeFilter();

};

#endif
