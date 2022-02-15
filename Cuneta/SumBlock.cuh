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
	int NumberOfLayers;
	int layerID;
	int levelID;

	SumBlock(int _height, int _width, int _numberOfLayers, int _layerID, int _levelID);
	SumBlock();
	void Sum(float** _inputSet_1, float** _inputSet_2);
	void DebugPrintAll();
	void PrintLayerParams();
};






#endif
