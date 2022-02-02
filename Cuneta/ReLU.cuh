#pragma once

#ifndef RELU_GPU_H
#define RELU_GPU_H

#include "cuda_runtime.h"
#include "CunetaModule.cuh"


class ReLU : public CunetaModule
{
public: 
	ReLU(float* _inputMatrix, float* _outputMatrix, int _inHeight, int _outHeight, int _inWidth, int _outWidth);
	ReLU(float* _inputMatrix, float* _outputMatrix);
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;

};

#endif
