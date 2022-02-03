#pragma once

#ifndef MAXPOOL_GPU_H
#define MAXPOOL_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class MaxPool : public CunetaModule
{
public:
	MaxPool(float* _inputMatrix, int _inputHeight, int _inputWidth);
	void ForwardPass() override;
	void BackwardPass() override;
	void UpdateModule() override;

	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
};

#endif
