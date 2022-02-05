#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
public:
	Convolution(int _filterSize, int _paddingSize);
	float* filter;
	float* flippedFilter;
	float* paddedBackpropInput;
	float* filterBackpropResult;
	int paddedInputHeight;
	int paddedInputWidth;
	int filterSize;
	int paddingSize;

	int hyperParam_Beta1;
	int hyperParam_Beta2;
	int hyperParam_Epsilon;
	int hyperParam_T;
	int hyperParam_alpha;

	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth);
	void UpdateModule() override;
	void Dialate(float* _input, float *_output);
	void InitializeFilter();
	void FlipFilter();
	void PadBackpropInput();

};

#endif
