
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "FolderManager.cuh"
#include "ImageIngester.cuh"
#include "Logger.cuh"
#include "Test_Utils.cuh"
using namespace std;
int main()
{

	cout << "Cuneta is starting..." << endl;
	int matrixHeight = 6;
	int matrixWidth = 4;
	string directory = "D:\\Documents\\Project Files\\Cuneta\\Test Files\\processed_data";
	string imageName = "Ingester_Ground_Truth_Test";

	ReLU test = ReLU();

	//CunetaFolderManager folderManager = CunetaFolderManager(directory);
	//folderManager.GetAllFoldersInDirectory();
	//CunetaLogger loggy = CunetaLogger();
	//ImageIngester ingester = ImageIngester();

	//for (int i = 0; i < folderManager.totalFolders; ++i)
	//{
	//	ingester.ReadData(folderManager.currentFolder, folderManager.currentImageName);
	//	loggy.AddImageNameToProcessingHistory(directory, folderManager.currentImageName);
	//	folderManager.OpenNextFolder();

	//}


	//int inputVectorizedSize = matrixHeight * matrixWidth;
	//float* fwdInput = new float[inputVectorizedSize];
	//float* backInput = new float[inputVectorizedSize];

	////Initialize dummy forward input;
	//int min = -1;
	//int max = 5;
	//int range = max - min + 1;

	//for (int i = 0; i < matrixHeight * matrixWidth; ++i)
	//{
	//	fwdInput[i] = rand() % range + min;
	//	std::cout << fwdInput[i] << std::endl;
	//}

	////Initialize backprop input;
	//min = 2;
	//max = 8;
	//range = max - min + 1;

	//for (int i = 0; i < matrixHeight * matrixWidth; ++i)
	//{
	//	backInput[i] = rand() % range + min;
	//	std::cout << backInput[i] << std::endl;
	//}

	//test.ForwardPass(fwdInput, matrixHeight, matrixWidth);
	//test.BackwardPass(backInput, matrixHeight, matrixWidth);
	
	//CunetaLogger loggy = CunetaLogger();
	//loggy.LogReLUState(test, directory, imageName, 1);
	//TestReLU(matrixWidth, matrixHeight, -1, 5, true); ///OK
	//TestBackpropReLU(matrixWidth, matrixHeight, -1, 5, 2, 8 ,true); ///OK

	//TestMaxPool(matrixWidth, matrixHeight, -1, 5, true); ///OK
	//TestBackpropMaxPool(matrixWidth, matrixHeight, -1, 5, 2, 8, true); ///OK

	//TestConvolution(matrixWidth, matrixHeight, -1, 5, 3, 2, true); ///OK
	//TestBackpropConvolution(matrixWidth, matrixHeight, -1, 5, 1,3 , 3, 2, true); ///OK


	//TestTransposeConvolution(matrixWidth, matrixHeight, -1, 5, 3, 2, true); ///OK
	//TestBackpropTransposeConvolution(matrixWidth, matrixHeight, -1, 5, 1,3 , 3, 2, true); ///OK

	TestBackpropErrorCalcModule(matrixHeight, matrixWidth, -1, 3, 0,1 , true);

	//TestImageIngester(inputPath, groundTruthPath, true);

	return 0;
}
