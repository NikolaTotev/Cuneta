#pragma once

#ifndef SUMBLOCK_GPU_H
#define SUMBLOCK_GPU_H

#include <crt/host_defines.h>


class SumBlock
{
public:
	float** InputSet_1;
	float** InputSet_2;
	float** Output;
	int Height;
	int Width;
	int NumberOfElements;

	SumBlock(int _height, int _width, int _numberOfElements);
	void Sum(float** _inputSet_1, float** _inputSet_2);
};






#endif
