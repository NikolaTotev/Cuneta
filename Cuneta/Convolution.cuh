#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
public:
	Convolution(int _filterSize, int _paddingSize);
	float* m_Filter;
	float* m_FlippedFilter;
	float* m_PaddedBackpropInput;
	float* m_FilterBackpropResult;
	float* m_AdamOptimizer_VMatrix;
	float* m_AdamOptimizer_SMatrix;
	float* m_AdamOptimizer_Corrected_VMatrix;
	float* m_AdamOptimizer_Corrected_SMatrix;
	int m_PaddedInputHeight;
	int m_PaddedInputWidth;
	int m_FilterSize;
	int m_PaddingSize;

	float m_HyperParam_Beta1;
	float m_HyperParam_Beta2;
	float m_HyperParam_Epsilon;
	int m_HyperParam_T;
	float m_HyperParam_alpha;

	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth);
	void UpdateModule() override;
	void Dialate(float* _input, float* _output);
	void InitializeFilter();
	void FlipFilter();
	void PadBackpropInput();
	void SetHyperParams(float _beta1, float _beta2, float _eps, int _t, float _alpha);
};

#endif
