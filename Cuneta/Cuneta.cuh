#include "Convolution.cuh"
#include "ErrorCalcModule.cuh"
#include "ImageIngester.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "Squishy.cuh"
#include "SumBlock.cuh"
#include "TransposeConvolution.cuh"


class Cuneta
{
public:
	Convolution Convolution_Level_1_Layer_1;
	ReLU ReLU_Level_1_Layer_1;
	Convolution Convolution_Level_1_Layer_2;
	ReLU ReLU_Level_1_Layer_2;

	MaxPool MaxPool_1;

	Convolution Convolution_Level_2_Layer_1;
	ReLU ReLU_Level_2_Layer_1;
	Convolution Convolution_Level_2_Layer_2;
	ReLU ReLU_Level_2_Layer_2;


	MaxPool MaxPool_2;

	Convolution Convolution_Level_3_Layer_1;
	ReLU ReLU_Level_3_Layer_1;
	Convolution Convolution_Level_3_Layer_2;
	ReLU ReLU_Level_3_Layer_2;

	MaxPool MaxPool_3;

	Convolution Convolution_Level_4_Layer_1;
	ReLU ReLU_Level_4_Layer_1;
	Convolution Convolution_Level_4_Layer_2;
	ReLU ReLU_Level_4_Layer_2;

	MaxPool MaxPool_4;

	Convolution Convolution_Level_5_Layer_1;
	ReLU ReLU_Level_5_Layer_1;
	Convolution Convolution_Level_5_Layer_2;
	ReLU ReLU_Level_5_Layer_2;

	Convolution SkipConnection_Convolution_1;
	TransposeConvolution TransposeConvolution_1;
	SumBlock SumBlock_1;
	
	Convolution Convolution_Level_6_Layer_1;
	ReLU ReLU_Level_6_Layer_1;

	Convolution SkipConnection_Convolution_2;
	TransposeConvolution TransposeConvolution_2;
	SumBlock SumBlock_2;

	Convolution Convolution_Level_7_Layer_1;
	ReLU ReLU_Level_7_Layer_1;

	Convolution SkipConnection_Convolution_3;
	TransposeConvolution TransposeConvolution_3;
	SumBlock SumBlock_3;

	Convolution Convolution_Level_8_Layer_1;
	ReLU ReLU_Level_8_Layer_1;
	
	Convolution SkipConnection_Convolution_4;
	TransposeConvolution TransposeConvolution_4;
	SumBlock SumBlock_4;

	Convolution Convolution_Level_9_Layer_1;
	ReLU ReLU_Level_9_Layer_1;

	Squishy Squoosh_1;

	//ErrorCalcModule Err_Module;

	int Input_HEIGHT;
	int Input_WIDTH;
	float** Input;
	Cuneta(int _height, int _width, float** _input);
	void PrintSetup();
	void Train(string dataDirectory, string logDirectory);
	void Segment(string inputImage, string logDirectory);
	void Save(string savePath);
};