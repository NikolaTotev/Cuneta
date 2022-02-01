#pragma once
#ifndef MODULE_H
#define MODULE_H

class CunetaModule
{
public:
	float* m_InputMatrix;
	float* m_OutputMatrix;

	int m_InputMatrixHeight;
	int m_InputMatrixWidth;

	int m_OutputMatrixHeight;
	int m_OutputMatrixWidth;

	virtual void ForwardPass() {};
	virtual void BackwardPass() {};
	virtual void UpdateModule() {};
};

#endif
