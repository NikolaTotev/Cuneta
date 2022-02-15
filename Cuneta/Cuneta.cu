#include "Cuneta.cuh"

#include <iostream>
using namespace std;

Cuneta::Cuneta(int _height, int _width, float** _input)
{
	Input_HEIGHT = _height;
	Input_WIDTH = _width;

	///FIRST CONVOLUTION LAYER =====================================================
	Convolution_Level_1_Layer_1 = Convolution(3, 2, 1, 16, Input_HEIGHT, Input_WIDTH, 1, 1);
	int Previous_NumberOf_Outputs = Convolution_Level_1_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	int Previous_Output_HEIGHT = Convolution_Level_1_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	int Previous_Output_WIDTH = Convolution_Level_1_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_1_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 1);
	Previous_NumberOf_Outputs = ReLU_Level_1_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_1_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_1_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_1_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 16, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 2);
	Previous_NumberOf_Outputs = Convolution_Level_1_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_1_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1,2);
	Previous_NumberOf_Outputs = Convolution_Level_1_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_1_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///FIRST MAXPOOL =====================================================
	MaxPool_1 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 1);
	Previous_NumberOf_Outputs = MaxPool_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_1.L_FORWARD_OutputLayer_WIDTH;

	///SECOND CONVOLUTION LAYER ======================================================
	Convolution_Level_2_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 32, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 1);
	Previous_NumberOf_Outputs = Convolution_Level_2_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_2_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_2_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_2_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 1);
	Previous_NumberOf_Outputs = ReLU_Level_2_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_2_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_2_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_2_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 32, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 2);
	Previous_NumberOf_Outputs = Convolution_Level_2_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_2_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_2_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_2_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 2);
	Previous_NumberOf_Outputs = ReLU_Level_2_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	///SECOND MAXPOOL ====================================================
	MaxPool_2 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 2);
	Previous_NumberOf_Outputs = MaxPool_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_2.L_FORWARD_OutputLayer_WIDTH;

	///THIRD CONVOLUTION LAYER =======================================================
	Convolution_Level_3_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 64, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 1);
	Previous_NumberOf_Outputs = Convolution_Level_3_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_3_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_3_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_3_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 1);
	Previous_NumberOf_Outputs = ReLU_Level_3_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_3_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_3_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_3_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 64, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 2);
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

	///FOURTH CONVOLUTION LAYER ======================================================
	Convolution_Level_4_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 1);
	Previous_NumberOf_Outputs = Convolution_Level_4_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_4_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_4_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_4_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 1);
	Previous_NumberOf_Outputs = ReLU_Level_4_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_4_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_4_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_4_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 2);
	Previous_NumberOf_Outputs = Convolution_Level_4_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_4_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_4_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_4_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 2);
	Previous_NumberOf_Outputs = ReLU_Level_4_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_WIDTH;   ///<<<< SKIP CONNECTION 1 COMES FROM HERE <<<<

	///FOURTH MAXPOOL ====================================================
	MaxPool_4 = MaxPool(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 4);
	Previous_NumberOf_Outputs = MaxPool_4.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = MaxPool_4.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = MaxPool_4.L_FORWARD_OutputLayer_WIDTH; 

	///FIFTH CONVOLUTION LAYER =======================================================
	Convolution_Level_5_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, 256, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 2);
	Previous_NumberOf_Outputs = Convolution_Level_5_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_5_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_5_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_5_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 5, 1);
	Previous_NumberOf_Outputs = ReLU_Level_5_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_5_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_5_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Convolution_Level_5_Layer_2 = Convolution(3, 2, Previous_NumberOf_Outputs, 256, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 2);
	Previous_NumberOf_Outputs = Convolution_Level_5_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_5_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_5_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_5_Layer_2 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 5, 1);
	Previous_NumberOf_Outputs = ReLU_Level_5_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_5_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_5_Layer_2.L_FORWARD_OutputLayer_WIDTH;


	///FIRST TRANSPOSE CONV CONNECTION =============================================
	TransposeConvolution_1 = TransposeConvolution(3, 2, Previous_NumberOf_Outputs, 128, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 1);
	Previous_NumberOf_Outputs = TransposeConvolution_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = TransposeConvolution_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = TransposeConvolution_1.L_FORWARD_OutputLayer_WIDTH;

	///FIRST SKIP CONNECTION =============================================

	int L4_Output_Count = ReLU_Level_4_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	int L4_Output_HEIGHT = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	int L4_Output_WIDTH = ReLU_Level_4_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	SkipConnection_Convolution_1 = Convolution(3, 2, L4_Output_Count, L4_Output_Count, L4_Output_HEIGHT, L4_Output_WIDTH, 10, 10);
	SumBlock_1 = SumBlock(L4_Output_HEIGHT, L4_Output_WIDTH, L4_Output_Count, 1, 1);


	///SIXTH CONVOLUTION LAYER =======================================================
	Convolution_Level_6_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 6, 1);
	Previous_NumberOf_Outputs = Convolution_Level_6_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_6_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_6_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_6_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 6, 1);
	Previous_NumberOf_Outputs = ReLU_Level_6_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_6_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_6_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	///SECOND TRANSPOSE CONV CONNECTION =============================================
	TransposeConvolution_2 = TransposeConvolution(3, 2, Previous_NumberOf_Outputs, 64, Previous_Output_HEIGHT, Previous_Output_WIDTH, 2, 1);
	Previous_NumberOf_Outputs = TransposeConvolution_2.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = TransposeConvolution_2.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = TransposeConvolution_2.L_FORWARD_OutputLayer_WIDTH;


	///SECOND SKIP CONNECTION =============================================

	int L3_Output_Count = ReLU_Level_3_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	int L3_Output_HEIGHT = ReLU_Level_3_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	int L3_Output_WIDTH = ReLU_Level_3_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	SkipConnection_Convolution_2 = Convolution(3, 2, L3_Output_Count, L3_Output_Count, L3_Output_HEIGHT, L3_Output_WIDTH, 2, 2);
	SumBlock_2 = SumBlock(L3_Output_HEIGHT, L3_Output_WIDTH, L3_Output_Count, 2, 2);

	///SEVENTH CONVOLUTION LAYER =======================================================
	Convolution_Level_7_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 7, 1);
	Previous_NumberOf_Outputs = Convolution_Level_7_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_7_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_7_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_7_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 7, 1);
	Previous_NumberOf_Outputs = ReLU_Level_7_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_7_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_7_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	///THIRD TRANSPOSE CONV CONNECTION =============================================
	TransposeConvolution_3 = TransposeConvolution(3, 2, Previous_NumberOf_Outputs, 32, Previous_Output_HEIGHT, Previous_Output_WIDTH, 3, 1);
	Previous_NumberOf_Outputs = TransposeConvolution_3.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = TransposeConvolution_3.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = TransposeConvolution_3.L_FORWARD_OutputLayer_WIDTH;

	///THIRD SKIP CONNECTION =============================================
	int L2_Output_Count = ReLU_Level_2_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	int L2_Output_HEIGHT = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	int L2_Output_WIDTH = ReLU_Level_2_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	SkipConnection_Convolution_3 = Convolution(3, 2, L2_Output_Count, L2_Output_Count, L2_Output_HEIGHT, L2_Output_WIDTH, 3, 3);
	SumBlock_3 = SumBlock(L2_Output_HEIGHT, L2_Output_WIDTH, L3_Output_Count, 30, 30);

	///EIGHTH CONVOLUTION LAYER =======================================================
	Convolution_Level_8_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 8, 1);
	Previous_NumberOf_Outputs = Convolution_Level_8_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_8_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_8_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_8_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 8, 1);
	Previous_NumberOf_Outputs = ReLU_Level_8_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_8_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_8_Layer_1.L_FORWARD_OutputLayer_WIDTH;


	///FOURTH TRANSPOSE CONV CONNECTION =============================================
	TransposeConvolution_4 = TransposeConvolution(3, 2, Previous_NumberOf_Outputs, 16, Previous_Output_HEIGHT, Previous_Output_WIDTH, 4, 1);
	Previous_NumberOf_Outputs = TransposeConvolution_4.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = TransposeConvolution_4.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = TransposeConvolution_4.L_FORWARD_OutputLayer_WIDTH;

	///FOURTH SKIP CONNECTION =============================================
	int L1_Output_Count = ReLU_Level_1_Layer_2.L_FORWARD_NumberOf_OUTPUTS;
	int L1_Output_HEIGHT = ReLU_Level_1_Layer_2.L_FORWARD_OutputLayer_HEIGHT;
	int L1_Output_WIDTH = ReLU_Level_1_Layer_2.L_FORWARD_OutputLayer_WIDTH;

	SkipConnection_Convolution_4 = Convolution(3, 2, L1_Output_Count, L1_Output_Count, L1_Output_HEIGHT, L1_Output_WIDTH, 4, 4);
	SumBlock_4 = SumBlock(L1_Output_HEIGHT, L1_Output_WIDTH, L1_Output_Count, 40, 40);


	///NINTH CONVOLUTION LAYER =======================================================
	Convolution_Level_9_Layer_1 = Convolution(3, 2, Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 9, 1);
	Previous_NumberOf_Outputs = Convolution_Level_9_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = Convolution_Level_9_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = Convolution_Level_9_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	ReLU_Level_9_Layer_1 = ReLU(Previous_NumberOf_Outputs, Previous_NumberOf_Outputs, Previous_Output_HEIGHT, Previous_Output_WIDTH, 9, 1);
	Previous_NumberOf_Outputs = ReLU_Level_9_Layer_1.L_FORWARD_NumberOf_OUTPUTS;
	Previous_Output_HEIGHT = ReLU_Level_9_Layer_1.L_FORWARD_OutputLayer_HEIGHT;
	Previous_Output_WIDTH = ReLU_Level_9_Layer_1.L_FORWARD_OutputLayer_WIDTH;

	Squoosh_1 = Squishy(1, 0, Previous_NumberOf_Outputs, 1, Previous_Output_HEIGHT, Previous_Output_WIDTH, 1, 1);

	///ERROR MODULE - Error module is setup in the "training function" for now.
}


void Cuneta::PrintSetup()
{
	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 1 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_1_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_1_Layer_1.PrintLayerParams();
	cout << endl;
	Convolution_Level_1_Layer_2.PrintLayerParams();
	cout << endl;
	ReLU_Level_1_Layer_2.PrintLayerParams();
	cout << endl;

	cout << "####################################" << endl;
	cout << "############ MAX POOl 1 ############" << endl;
	cout << "####################################" << endl;
	MaxPool_1.PrintLayerParams();

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 2 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_2_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_2_Layer_1.PrintLayerParams();
	cout << endl;
	Convolution_Level_2_Layer_2.PrintLayerParams();
	cout << endl;
	ReLU_Level_2_Layer_2.PrintLayerParams();
	cout << endl;


	cout << "####################################" << endl;
	cout << "############ MAX POOl 2 ############" << endl;
	cout << "####################################" << endl;

	MaxPool_2.PrintLayerParams();

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 3 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_3_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_3_Layer_1.PrintLayerParams();
	cout << endl;
	Convolution_Level_3_Layer_2.PrintLayerParams();
	cout << endl;
	ReLU_Level_3_Layer_2.PrintLayerParams();
	cout << endl;

	cout << "####################################" << endl;
	cout << "############ MAX POOl 3 ############" << endl;
	cout << "####################################" << endl;
	MaxPool_3.PrintLayerParams();

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 4 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_4_Layer_1.PrintLayerParams();
	ReLU_Level_4_Layer_1.PrintLayerParams();
	cout << endl;
	Convolution_Level_4_Layer_2.PrintLayerParams();
	cout << endl;
	ReLU_Level_4_Layer_2.PrintLayerParams();
	cout << endl;

	cout << "####################################" << endl;
	cout << "############ MAX POOl 4 ############" << endl;
	cout << "####################################" << endl;
	MaxPool_4.PrintLayerParams();

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 5 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_5_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_5_Layer_1.PrintLayerParams();
	cout << endl;
	Convolution_Level_5_Layer_2.PrintLayerParams();
	cout << endl;
	ReLU_Level_5_Layer_2.PrintLayerParams();
	cout << endl;

	cout << "=====================================================" << endl;
	cout << "================= SKIP CONNECTION 1 ================= " << endl;
	cout << "=====================================================" << endl;
	SkipConnection_Convolution_1.PrintLayerParams();
	cout << endl;
	TransposeConvolution_1.PrintLayerParams();
	cout << endl;
	SumBlock_1.PrintLayerParams();
	cout << endl;

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 6 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_6_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_6_Layer_1.PrintLayerParams();
	cout << endl;

	cout << "=====================================================" << endl;
	cout << "================= SKIP CONNECTION 2 ================= " << endl;
	cout << "=====================================================" << endl;
	SkipConnection_Convolution_2.PrintLayerParams();
	cout << endl;
	TransposeConvolution_2.PrintLayerParams();
	cout << endl;
	SumBlock_2.PrintLayerParams();
	cout << endl;

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 7 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_7_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_7_Layer_1.PrintLayerParams();
	cout << endl;

	cout << "=====================================================" << endl;
	cout << "================= SKIP CONNECTION 3 ================= " << endl;
	cout << "=====================================================" << endl;
	SkipConnection_Convolution_3.PrintLayerParams();
	cout << endl;
	TransposeConvolution_3.PrintLayerParams();
	cout << endl;
	SumBlock_3.PrintLayerParams();
	cout << endl;

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 8 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_8_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_8_Layer_1.PrintLayerParams();
	cout << endl;

	cout << "=====================================================" << endl;
	cout << "================= SKIP CONNECTION 4 ================= " << endl;
	cout << "=====================================================" << endl;
	SkipConnection_Convolution_4.PrintLayerParams();
	cout << endl;
	TransposeConvolution_4.PrintLayerParams();
	cout << endl;
	SumBlock_4.PrintLayerParams();
	cout << endl;

	cout << "================================================" << endl;
	cout << "================= CONV LEVEL 9 ================= " << endl;
	cout << "================================================" << endl;
	Convolution_Level_9_Layer_1.PrintLayerParams();
	cout << endl;
	ReLU_Level_9_Layer_1.PrintLayerParams();
	cout << endl;


	Squoosh_1.PrintLayerParams();


}
