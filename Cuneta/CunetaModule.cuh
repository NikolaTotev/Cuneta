#pragma once
#ifndef MODULE_H
#define MODULE_H

class CunetaModule
{
public:
	float* m_InputMatrix;
	float* m_BackPropInputMatrix;
	float* m_BackpropagationOutput;
	float* m_OutputMatrix;


	//Variables for the full layerSize
	float** L_FORWARD_Pass_INPUTS;
	float** L_FORWARD_Pass_OUTPUTS;

	float** L_BACKWARD_Pass_INPUTS;
	float** L_BACKWARD_Pass_OUTPUTS;
	float** L_Filters;
	float* L_Baises;

	int L_NumberOf_FILTERS;

	int L_FORWARD_NumberOf_INPUTS;
	int L_FORWARD_NumberOf_OUTPUTS;

	int L_BACKWARD_NumberOf_INPUTS;
	int L_BACKWARD_NumberOf_OUTPUTS;

	//Backward Params
	int L_FORWARD_InputLayer_HEIGHT;
	int L_FORWARD_InputLayer_WIDTH;

	int L_FORWARD_OutputLayer_HEIGHT;
	int L_FORWARD_OutputLayer_WIDTH;

	//Forward params
	int L_BACKWARD_InputLayer_HEIGHT;
	int L_BACKWARD_InputLayer_WIDTH;

	int L_BACKWARD_OutputLayer_HEIGHT;
	int L_BACKWARD_OutputLayer_WIDTH;

	int m_InputMatrixHeight;
	int m_InputMatrixWidth;

	int m_OutputMatrixHeight;
	int m_OutputMatrixWidth;

	int m_BackpropInputMatrixHeight;
	int m_BackpropInputMatrixWidth;

	int m_BackpropOutputMatrixHeight;
	int m_BackpropOutputMatrixWidth;

	virtual void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) {};
	virtual void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) {};

	virtual void LayerForwardPass(float** _inputs) {};
	virtual void LayerBackwardPass(float** _backpropInput) {};

	virtual void UpdateModule() {};
};

#endif
