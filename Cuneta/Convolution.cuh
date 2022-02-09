#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"


class Convolution : public CunetaModule
{
public:
	Convolution(int _filterSize, int _paddingSize, int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth);
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

	float* testIt;

	int L_BACKWARD_InputLayer_PADDED_HEIGHT;
	int L_BACKWARD_InputLayer_PADDED_WIDTH;

	float** L_BACKWARD_Pass_PADDED_INPUTS;



	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth);
	void LayerForwardPass(float** _inputs) override;
	void LayerBackwardPass(float** _backpropInput) override;
	void UpdateModule() override;
	void Dialate(float* _input, float* _output);
	void InitializeFilter();
	void LayerFilterInitialization();
	void FlipFilter();
	void LayerFlipFilter();
	void PadBackpropInput();
	void LayerPadBackpropInput();
	void SetHyperParams(float _beta1, float _beta2, float _eps, int _t, float _alpha);
};

#endif
