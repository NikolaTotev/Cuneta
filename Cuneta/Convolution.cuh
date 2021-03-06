#pragma once

#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H

#include <crt/host_defines.h>

#include "CunetaModule.cuh"
using namespace std;
#include  <string>

class Convolution : public CunetaModule
{
public:
	Convolution(int _filterSize, int _paddingSize, int _numberOfInputs, int _numberOfOutputs, int _inputHeight, int _inputWidth, int _layerID, int _levelID);
	Convolution();
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

	int L_BACKWARD_InputLayer_PADDED_HEIGHT;
	int L_BACKWARD_InputLayer_PADDED_WIDTH;

	float** L_BACKWARD_Pass_PADDED_INPUTS;
	float** L_FLIPPED_Filters;
	float** L_Filter_BACKPROP_RESULTS;	 

	float** L_AdamOptimizer_V_Matrix;
	float** L_AdamOptimizer_S_Matrix;
	float** L_AdamOptimizer_Corrected_V_Matrix;
	float** L_AdamOptimizer_Corrected_S_Matrix;

	float** L_Biases;
	float** L_PrevBiases;
	float** L_BIAS_AdamOptimizer_V_Matrix;
	float** L_BIAS_AdamOptimizer_S_Matrix;
	float** L_BIAS_AdamOptimizer_Corrected_V_Matrix;
	float** L_BIAS_AdamOptimizer_Corrected_S_Matrix;


	void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) override;
	void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) override;
	void FilterBackprop(float* backpropInput, int backPassHeight, int backPassWidth);
	void LayerForwardPass(float** _inputs) override;
	void LayerBackwardPass(float** _backpropInput) override;
	void LayerFilterBackprop();
	void LayerBiasUpdate();
	void LayerFilterInitialization();
	void LayerBiasInitialization();
	void LayerFlipFilter();
	void LayerPadBackpropInput();
	void LayerUpdate();
	void UpdateModule() override;
	void Dialate(float* _input, float* _output);
	void InitializeFilter();
	void FlipFilter();
	void PadBackpropInput();
	void SetHyperParams(float _beta1, float _beta2, float _eps, int _t, float _alpha);
	void Print();
	void DebugPrintAll();
	void PrintLayerParams();
	void CudaErrHandler(string responsibleModule, string function);

};

#endif
