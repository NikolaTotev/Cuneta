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

	int m_InputMatrixHeight;
	int m_InputMatrixWidth;

	int m_BackpropInputMatrixHeight;
	int m_BackpropInputMatrixWidth;

	int m_OutputMatrixHeight;
	int m_OutputMatrixWidth;

	int m_BackpropOutputMatrixHeight;
	int m_BackpropOutputMatrixWidth;

	virtual void ForwardPass(float* forwardPassInput, int fwdPassHeight, int fwdPassWidth) {};
	virtual void BackwardPass(float* backpropInput, int backPassHeight, int backPassWidth) {};
	virtual void UpdateModule() {};
};

#endif
