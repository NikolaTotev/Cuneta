#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
public:
	Convolution(int _filterSize);
	float* filter;
	int filterSize;
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void UpdateModule() override;
	void Dialate(float* _input, float *_output);
	void InitializeFilter();

};

#endif
