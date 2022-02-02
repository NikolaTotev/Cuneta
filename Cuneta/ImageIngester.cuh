#pragma once
#include <string>

#ifndef IMAGE_INGESTER_GPU_H
#define IMAGE_INGESTER_GPU_H

#include <crt/host_defines.h>


class ImageIngester
{
public:
	std::string baseDirectory;
	float* rawImageData;
	float* normalizedImageData;
	ImageIngester(std::string _baseDir);

	void ReadRawImageData();
	void Normalize();

	__global__ void NormalizationKernel(float* d_Input, float* d_Output);

};

#endif
