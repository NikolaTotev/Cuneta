#include "Cuneta.cuh"


Cuneta::Cuneta(int _height, int _width, float**_input)
{
	Input_HEIGHT = _height;
	Input_WIDTH = _width;

	///FIRST LAYER =====================================================
	Convolution_Level_1_Layer_1 = Convolution(3,2,1, 16, Input_HEIGHT, Input_WIDTH);
	int Previous_NumberOf_Outputs = Convolution_Level_1_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	int Previous_Output_HEIGHT = Convolution_Level_1_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	int Previous_Output_WIDTH= Convolution_Level_1_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_1_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1,1);
	Previous_NumberOf_Outputs = ReLU_Level_1_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_1_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_1_Layer_1.L_FORWARD_OutputLayer_WIDTH;


	Convolution_Level_1_Layer_2 = Convolution(3,2,Previous_NumberOf_Outputs, 16, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_1_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///FIRST MAXPOOL =====================================================
	MaxPool_1 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 1);
	Previous_NumberOf_Outputs = MaxPool_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_1.L_FORWARD_OutputLayer_WIDTH;

	///SECOND LAYER ======================================================
	Convolution_Level_2_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 32, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_2_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_2_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_2_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_2_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 1);
	Previous_NumberOf_Outputs = ReLU_Level_2_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_2_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_2_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_2_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 32, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_2_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_2_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_2_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_2_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 2);
	Previous_NumberOf_Outputs = ReLU_Level_2_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///SECOND MAXPOOL ====================================================
	MaxPool_2 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2,2);

	///THIRD LAYER =======================================================
	Convolution_Level_3_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 64, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_3_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_3_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_3_Layer_1.L_FORWARD_OutputLayer_WIDTH;


	ReLU_Level_3_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3,1);
	Previous_NumberOf_Outputs = ReLU_Level_3_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_3_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_3_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_3_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 64, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_3_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_3_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_3_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_3_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 2);
	Previous_NumberOf_Outputs = ReLU_Level_3_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_3_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_3_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///THIRD MAXPOOL =====================================================
	MaxPool_3 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 3);
	Previous_NumberOf_Outputs = MaxPool_3.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_3.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_3.L_FORWARD_OutputLayer_WIDTH;
	
	///FOURTH LAYER ======================================================
	Convolution_Level_4_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_4_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_4_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_4_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_4_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 1);
	Previous_NumberOf_Outputs = ReLU_Level_4_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_4_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_4_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_4_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_4_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_4_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_4_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_4_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH,4,2);
	Previous_NumberOf_Outputs = ReLU_Level_4_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///FOURTH MAXPOOL ====================================================
	MaxPool_4 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 4);
	Previous_NumberOf_Outputs = MaxPool_4.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_4.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_4.L_FORWARD_OutputLayer_WIDTH;
	
	///FIFTH LAYER =======================================================
	Convolution_Level_5_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 256, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	Previous_NumberOf_Outputs = Convolution_Level_5_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_5_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_5_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	///FIRST SKIP CONNECTION =============================================
	TransposeConvolution_1 = TransposeConvolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH);
	int SkipFrom_Level4_
















}
