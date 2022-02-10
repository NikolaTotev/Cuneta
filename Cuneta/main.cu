
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

	TransposeConvolution tconv = TransposeConvolution(3, 2, 4, 2, 4, 6);
	tconv.LayerFilterInitialization();
	int counter = 1;

	float** back_inputs = new float* [2];

	for (int j = 0; j < 2; ++j)
	{
		back_inputs[j] = new float[8 * 6];

		for (int i = 0; i < 8 * 6; ++i)
		{
			back_inputs[j][i] = (j);
		}
	}

	float** inputs = new float* [4];

	for (int j = 0; j < 4; ++j)
	{
		inputs[j] = new float[4 * 6];

		for (int i = 0; i < 4 * 6; ++i)
		{
			inputs[j][i] = j + 1;
		}
	}

	tconv.LayerForwardPass(inputs);
	tconv.LayerBackwardPass(back_inputs);
	tconv.LayerFilterBackprop();

	cout << endl;
	cout << endl;
	cout << "============ BACKWARD INPUTS ============" << endl;

	for (int j = 0; j < tconv.L_BACKWARD_NumberOf_INPUTS; ++j)
	{

		for (int i = 0; i < tconv.L_BACKWARD_InputLayer_HEIGHT * tconv.L_BACKWARD_InputLayer_WIDTH; ++i)
		{
			cout << tconv.L_BACKWARD_Pass_INPUTS[j][i] << " ";
			counter++;
			if (counter == tconv.L_BACKWARD_InputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;
	}



	cout << endl;
	cout << endl;
	cout << "============ FORWARD PADDED INPUTS ============" << endl;

	for (int j = 0; j < tconv.L_FORWARD_NumberOf_INPUTS; ++j)
	{

		for (int i = 0; i < tconv.L_FORWARD_InputLayer_PADDED_HEIGHT* tconv.L_FORWARD_InputLayer_PADDED_WIDTH; ++i)
		{
			cout << tconv.L_FORWARD_Pass_PADDED_INPUTS[j][i] << " ";
			counter++;
			if (counter == tconv.L_FORWARD_InputLayer_PADDED_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;
	}
	cout << endl;
	cout << endl;

	cout << "============ FILTER BACKPROP OUTPUTS ============" << endl;

	for (int j = 0; j < tconv.L_NumberOf_FILTERS; ++j)
	{

		for (int i = 0; i < tconv.m_FilterSize * tconv.m_FilterSize; ++i)
		{
			cout << tconv.L_Filter_BACKPROP_RESULTS[j][i] << " ";
			counter++;
			if (counter == tconv.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;

	}
		

	/*int counter = 1;

	float** back_inputs = new float* [2];

	for (int j = 0; j < 2; ++j)
	{
		back_inputs[j] = new float[10 * 8];
		
		for (int i = 0; i < 10 * 8; ++i)
		{
			back_inputs[j][i] = (j+1);
		}
	}

	float** inputs = new float* [4];

	for (int j = 0; j < 4; ++j)
	{
		inputs[j] = new float[4 * 8];

		for (int i = 0; i < 4 * 8; ++i)
		{
			inputs[j][i] = j + 1;
		}
	}

	tconv.LayerForwardPass(inputs);
	tconv.LayerBackwardPass(back_inputs);

	cout << endl;
	cout << endl;
	cout << "============ BACKWARD INPUTS ============" << endl;

	for (int j = 0; j < tconv.L_BACKWARD_NumberOf_INPUTS; ++j)
	{

		for (int i = 0; i < tconv.L_BACKWARD_InputLayer_HEIGHT * tconv.L_BACKWARD_InputLayer_WIDTH; ++i)
		{
			cout << tconv.L_BACKWARD_Pass_INPUTS[j][i] << " ";
			counter++;
			if (counter == tconv.L_BACKWARD_InputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;
	}

	cout << endl;
	cout << endl;
	cout << "============ FILTER INPUTS ============" << endl;

	for (int j = 0; j < tconv.L_NumberOf_FILTERS; ++j)
	{

		for (int i = 0; i < tconv.m_FilterSize * tconv.m_FilterSize; ++i)
		{
			cout << tconv.L_Filters[j][i] << " ";
			counter++;
			if (counter == tconv.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;

	}

	cout << endl;
	cout << endl;
	cout << "============FLIPPED FILTER INPUTS ============" << endl;

	for (int j = 0; j < tconv.L_NumberOf_FILTERS; ++j)
	{

		for (int i = 0; i < tconv.m_FilterSize * tconv.m_FilterSize; ++i)
		{
			cout << tconv.L_FLIPPED_Filters[j][i] << " ";
			counter++;
			if (counter == tconv.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;

	}

	cout << endl;
	cout << endl;
	cout << "============ BACKWARD OUTPUTS ============" << endl;

	for (int j = 0; j < tconv.L_BACKWARD_NumberOf_OUTPUTS; ++j)
	{

		for (int i = 0; i < tconv.L_BACKWARD_OutputLayer_HEIGHT * tconv.L_BACKWARD_OutputLayer_WIDTH; ++i)
		{
			cout << tconv.L_BACKWARD_Pass_OUTPUTS[j][i] << " ";
			counter++;
			if (counter == tconv.L_BACKWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;
	}*/


	//int counter = 1;

	//float** forward_inputs = new float* [4];

	//for (int j = 0; j < 4; ++j)
	//{
	//	forward_inputs[j] = new float[8 * 4];

	//	for (int i = 0; i < 8 * 4; ++i)
	//	{
	//		forward_inputs[j][i] = (j);
	//	}
	//}

	//float** inputs = new float* [4];

	//for (int j = 0; j < 4; ++j)
	//{
	//	inputs[j] = new float[2 * 6];

	//	for (int i = 0; i < 2 * 6; ++i)
	//	{
	//		inputs[j][i] = j + 1;
	//	}
	//}
	//tconv.LayerForwardPass(forward_inputs);

	//cout << endl;
	//cout << endl;
	//cout << "============ FORWARD INPUTS ============" << endl;

	//for (int j = 0; j < tconv.L_FORWARD_NumberOf_INPUTS; ++j)
	//{

	//	for (int i = 0; i < tconv.L_FORWARD_InputLayer_HEIGHT * tconv.L_FORWARD_InputLayer_WIDTH; ++i)
	//	{
	//		cout << tconv.L_FORWARD_Pass_INPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == tconv.L_FORWARD_InputLayer_WIDTH + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;
	//}

	//cout << endl;
	//cout << endl;
	//cout << "============ PADDED FORWARD INPUTS ============" << endl;

	//for (int j = 0; j < tconv.L_FORWARD_NumberOf_INPUTS; ++j)
	//{
	//	
	//	for (int i = 0; i < tconv.L_FORWARD_InputLayer_PADDED_HEIGHT * tconv.L_FORWARD_InputLayer_PADDED_WIDTH; ++i)
	//	{
	//		cout << tconv.L_FORWARD_Pass_PADDED_INPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == tconv.L_FORWARD_InputLayer_PADDED_WIDTH + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;
	//}

	//cout << endl;
	//cout << endl;
	//cout << "============ FILTER INPUTS ============" << endl;

	//for (int j = 0; j < tconv.L_NumberOf_FILTERS; ++j)
	//{

	//	for (int i = 0; i < tconv.m_FilterSize * tconv.m_FilterSize; ++i)
	//	{
	//		cout << tconv.L_Filters[j][i] << " ";
	//		counter++;
	//		if (counter == tconv.m_FilterSize + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;

	//}

	//cout << endl;
	//cout << endl;
	//cout << "============ FORWARD OUTPUTS ============" << endl;

	//for (int j = 0; j < tconv.L_FORWARD_NumberOf_OUTPUTS; ++j)
	//{

	//	for (int i = 0; i < tconv.L_FORWARD_OutputLayer_HEIGHT * tconv.L_FORWARD_OutputLayer_WIDTH; ++i)
	//	{
	//		cout << tconv.L_FORWARD_Pass_OUTPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == tconv.L_FORWARD_OutputLayer_WIDTH + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;
	//}



	//Convolution conv = Convolution(3, 2, 2, 4, 8, 4);
	//conv.LayerFilterInitialization();

	//int counter = 1;

	//float** forward_inputs = new float* [4];

	//for (int j = 0; j < 4; ++j)
	//{
	//	forward_inputs[j] = new float[8*4];

	//	for (int i = 0; i < 8*4; ++i)
	//	{
	//		forward_inputs[j][i] = (j);
	//	}
	//}

	//float** inputs = new float* [4];

	//for (int j = 0; j < 4; ++j)
	//{
	//	inputs[j] = new float[2 * 6];

	//	for (int i = 0; i < 2 * 6; ++i)
	//	{
	//		inputs[j][i] = j + 1;
	//	}
	//}
	//conv.LayerForwardPass(forward_inputs);
	//conv.LayerBackwardPass(inputs);
	//conv.LayerFilterBackprop();
	//
	//cout << endl;
	//cout << endl;
	//cout << "============ FORWARD INPUTS ============" << endl;

	//for (int j = 0; j < conv.L_FORWARD_NumberOf_INPUTS; ++j)
	//{

	//	for (int i = 0; i < conv.L_FORWARD_InputLayer_HEIGHT*conv.L_FORWARD_InputLayer_WIDTH; ++i)
	//	{
	//		cout << conv.L_FORWARD_Pass_INPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == conv.L_FORWARD_InputLayer_WIDTH + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;
	//}

	//cout << endl;
	//cout << endl;
	//cout << "============ BACKWARD INPUTS ============" << endl;

	//for (int j = 0; j < conv.L_BACKWARD_NumberOf_INPUTS; ++j)
	//{

	//	for (int i = 0; i < conv.L_BACKWARD_InputLayer_HEIGHT * conv.L_BACKWARD_InputLayer_WIDTH; ++i)
	//	{
	//		cout << conv.L_BACKWARD_Pass_INPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == conv.L_BACKWARD_InputLayer_WIDTH + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;
	//}

	//cout << endl;
	//cout << endl;
	//cout << "============ BACKWARD INPUTS ============" << endl;

	//for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	//{

	//	for (int i = 0; i < conv.m_FilterSize * conv.m_FilterSize; ++i)
	//	{
	//		cout << conv.L_Filter_BACKPROP_RESULTS[j][i] << " ";
	//		counter++;
	//		if (counter == conv.m_FilterSize + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//	cout << endl;

	//}
	/*cout << endl;
	cout << endl;
	cout << "============ PADDED BACKPROP INPUTS ============" << endl;

	for (int j = 0; j < 4; ++j)
	{

		for (int i = 0; i < conv.L_BACKWARD_InputLayer_PADDED_HEIGHT*conv.L_BACKWARD_InputLayer_PADDED_WIDTH; ++i)
		{
			cout << conv.L_BACKWARD_Pass_PADDED_INPUTS[j][i] << " ";
			counter++;
			if (counter == conv.L_BACKWARD_InputLayer_PADDED_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}

		cout << endl;
		cout << endl;
	}


	cout << endl;
	cout << endl;
	cout << "============ BACKPROP FILTERS ============" << endl;

	for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	{

		for (int i = 0; i < conv.m_FilterSize * conv.m_FilterSize; ++i)
		{
			cout << conv.L_FLIPPED_Filters[j][i] << " ";
			counter++;
			if (counter == conv.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
		cout << endl;
	}

	cout << endl;
	cout << endl;
	cout << "============ BACKPROP OUTPUTS ============" << endl;

	for (int j = 0; j < conv.L_BACKWARD_NumberOf_OUTPUTS; ++j)
	{

		for (int i = 0; i < conv.L_BACKWARD_OutputLayer_HEIGHT * conv.L_BACKWARD_OutputLayer_WIDTH; ++i)
		{
			cout << conv.L_BACKWARD_Pass_OUTPUTS[j][i] << " ";
			counter++;
			if (counter == conv.L_BACKWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}

		cout << endl;
		cout << endl;
	}*/

	//inputs[0] = new float[2 * 3];

	//for (int i = 0; i < 4 * 6; ++i)
	//{
	//	inputs[0][i] = 1;
	//}

	//inputs[1] = new float[2 * 3];

	//for (int i = 0; i < 4 * 6; ++i)
	//{
	//	inputs[1][i] = 2;
	//}


	//conv.LayerForwardPass(inputs);



	//cout << "First input" << endl;
	//for (int i = 0; i < 24; ++i)
	//{
	//	cout << conv.L_FORWARD_Pass_INPUTS[0][i] << " ";
	//	counter++;
	//	if (counter == 4 + 1)
	//	{
	//		cout << endl;
	//		counter = 1;
	//	}
	//}
	//cout << endl;
	//cout << "Second input" << endl;

	//for (int i = 0; i < 24; ++i)
	//{
	//	cout << conv.L_FORWARD_Pass_INPUTS[1][i] << " ";
	//	counter++;
	//	if (counter == 4 + 1)
	//	{
	//		cout << endl;
	//		counter = 1;
	//	}
	//}

	//cout << endl;
	//cout << endl;
	//cout << "=================================" << endl;
	//cout << "=================================" << endl;
	//cout << "============ Filters ============" << endl;
	//cout << "=================================" << endl;
	//cout << "Number of filters: " << conv.L_NumberOf_FILTERS << endl;
	//for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	//{
	//	cout << "Filter " << j << endl;

	//	for (int i = 0; i < 9; ++i)
	//	{
	//		cout << conv.L_Filters[j][i] << " ";
	//		counter++;
	//		if (counter == 3 + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//}

	//cout << endl;
	//cout << endl;
	//cout << "=================================" << endl;
	//cout << "=================================" << endl;
	//cout << "============ Outputs ============" << endl;
	//cout << "=================================" << endl;

	//for (int j = 0; j < 4; ++j)
	//{
	//	cout << "Output " << j << endl;

	//	for (int i = 0; i < 6; ++i)
	//	{
	//		cout << conv.L_FORWARD_Pass_OUTPUTS[j][i] << " ";
	//		counter++;
	//		if (counter == 2 + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;

	//}


	//cout << endl;
	//cout << endl;
	//cout << "=================================" << endl;
	//cout << "=================================" << endl;
	//cout << "============ Filters ============" << endl;
	//cout << "=================================" << endl;
	//cout << "Number of filters: " << conv.L_NumberOf_FILTERS << endl;
	//for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	//{
	//	cout << "Filter " << j << endl;

	//	for (int i = 0; i < 9; ++i)
	//	{
	//		cout << conv.L_Filters[j][i] << " ";
	//		counter++;
	//		if (counter == 3 + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//}

	//conv.LayerFlipFilter();

	//cout << "============ AFTER FLIP ============" << endl;

	//cout << endl;
	//cout << endl;
	//cout << "=================================" << endl;
	//cout << "=================================" << endl;
	//cout << "============ Filters ============" << endl;
	//cout << "=================================" << endl;
	//cout << "Number of filters: " << conv.L_NumberOf_FILTERS << endl;
	//for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	//{
	//	cout << "Filter " << j << endl;

	//	for (int i = 0; i < 9; ++i)
	//	{
	//		cout << conv.L_Filters[j][i] << " ";
	//		counter++;
	//		if (counter == 3 + 1)
	//		{
	//			cout << endl;
	//			counter = 1;
	//		}
	//	}
	//	cout << endl;
	//}

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

	//TestBackpropErrorCalcModule(matrixHeight, matrixWidth, -1, 3, 0, 1, true);

	//TestImageIngester(inputPath, groundTruthPath, true);

	return 0;
}
