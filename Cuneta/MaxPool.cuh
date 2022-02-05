#pragma once

#ifndef MAXPOOL_GPU_H
#define MAXPOOL_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class MaxPool : public CunetaModule
{
public:
	MaxPool();
	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void UpdateModule() override;

	//Input and output will be in global memory. d_ shows in which memory the variables are stored.
};

#endif
