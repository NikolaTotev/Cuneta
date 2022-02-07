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

	ImageIngester();
	void ReadData(string _currentFolder, string _currentImageName);
		
};

#endif
