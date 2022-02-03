
#include <algorithm>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Test_Utils.cuh"

int main()
{
	int matrixHeight = 6;
	int matrixWidth = 6;
	int vectorizedMatrixSize = matrixWidth * matrixHeight;
	float* input = new float[vectorizedMatrixSize];
	float* output = new float[vectorizedMatrixSize];

	int max = 20;
	int min = -5;
	int range = max - min + 1;
	
	for (int i = 0; i < matrixHeight*matrixWidth; ++i)
	{
		input[i] = rand() % range + min;
	}

	//TestReLU(input, output, matrixWidth, matrixHeight);

	TestMaxPool(input, output, matrixWidth, matrixHeight);

	return 0;
}
