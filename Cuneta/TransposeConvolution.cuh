#pragma once

#ifndef TRANSPOSE_CONVOLUTION_GPU_H
#define TRANSPOSE_CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class TransposeConvolution : public CunetaModule
{
public:
	TransposeConvolution(float* _inputMatrix, int _inputHeight, int _inputWidth, int _filterSize, int _paddingSize);
	float* filter;
	float* paddedInput;
	int filterSize;
	int paddingSize; 
	int paddedInputHeight; 
	int paddedInputWidth; 
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;
	void PadInput();
	void InitializeFilter();
	

};

#endif
