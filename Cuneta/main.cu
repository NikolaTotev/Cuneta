
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Test_Utils.cuh"
using namespace std;
int main()
{

	cout << "Cuneta is starting..." << endl;
	int matrixHeight = 4;
	int matrixWidth = 4;
	int vectorizedMatrixSize = matrixWidth * matrixHeight;
	float* input = new float[vectorizedMatrixSize];
	float* output = new float[vectorizedMatrixSize];

	int max = 20;
	int min = -5;
	int range = max - min + 1;

	for (int i = 0; i < matrixHeight * matrixWidth; ++i)
	{
		input[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	//TestReLU(input, output, matrixWidth, matrixHeight);

	/*std::cout << input[0] << std::endl;
	std::cout << input[1] << std::endl;
	std::cout << input[matrixWidth] << std::endl;
	std::cout << input[matrixWidth + 1] << std::endl;
	TestMaxPool(input, output, matrixWidth, matrixHeight, false);*/

	/*std::cout << input[0] << std::endl;
	std::cout << input[1] << std::endl;
	std::cout << input[2] << std::endl;
	std::cout << input[matrixWidth] << std::endl;
	std::cout << input[matrixWidth + 1] << std::endl;
	std::cout << input[matrixWidth + 2] << std::endl;*/

	TestConvolution(input, matrixHeight, matrixWidth, 3,true);

	//TestTransposeConvolution(input, matrixHeight, matrixWidth, 3, true);

	return 0;
}
