
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
	int matrixHeight = 6;
	int matrixWidth = 4;


	//TestReLU(matrixWidth, matrixHeight, -1, 5, true); ///OK
	//TestBackpropReLU(matrixWidth, matrixHeight, -1, 5, 2, 8 ,true); ///OK

	TestMaxPool(matrixWidth, matrixHeight, -1, 5, true); ///OK
	TestBackpropMaxPool(matrixWidth, matrixHeight, -1, 5, 2, 8, true); ///OK

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

	//TestConvolution(input, matrixHeight, matrixWidth, 3, true);

	//TestTransposeConvolution(input, matrixHeight, matrixWidth, 3, true);
	//TestErrorCalcModule(input, groundTruth, matrixHeight, matrixWidth, true);

	//TestImageIngester(inputPath, groundTruthPath, true);

	return 0;
}
