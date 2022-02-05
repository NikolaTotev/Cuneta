#pragma once
#include <string>

#ifndef IMAGE_INGESTER_GPU_H
#define IMAGE_INGESTER_GPU_H

using namespace std;

class ImageIngester
{
public:
	string inputPath;
	string groundTruthPath;
	int inputHeight;
	int inputWidth;

	int groundTruthHeight;
	int groundTruthWidth;
	float* inputImageData;
	float* groundTruthData;

	ImageIngester(string _inputPath, string _groundTruthPath);
	void ReadData();
		
};

#endif
