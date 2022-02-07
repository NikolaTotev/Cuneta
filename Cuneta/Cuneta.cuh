#include "Convolution.cuh"
#include "ErrorCalcModule.cuh"
#include "ImageIngester.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "TransposeConvolution.cuh"


class Cuneta
{
	void Train(string dataDirectory, string logDirectory);
	void Segment(string inputImage, string logDirectory);
	void Save(string savePath);
};