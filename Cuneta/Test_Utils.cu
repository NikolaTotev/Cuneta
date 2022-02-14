#include "Test_Utils.cuh"

#include <iostream>

#include "Squishy.cuh"
#include "SumBlock.cuh"


using namespace std;

NetworkValidator::NetworkValidator(string _saveDirectory)
{
	testSaveDirectory = _saveDirectory;
}

void NetworkValidator::TestFlowController()
{
	string userInput;
	cout << ">>>> Testing mode is now active <<<< " << endl;
	cout << "Test results will be saved here: " << endl << testSaveDirectory << endl;
	cout << "To exit from the testing mode, just enter 'stop'." << endl;
	cout << "If an test is executing wait for it to finish then enter the next command." << endl;

	cout << "The available commands are:" << endl;
	cout << "RELU" << endl;
	cout << "MAX" << endl;
	cout << "CONV" << endl;
	cout << "TCONV" << endl;
	cout << "SQUISH" << endl;
	cout << "ERR" << endl;
	cout << "SUMBLOCK" << endl;
	cout << "ALL" << endl;

	cout << "What is your first command?" << endl;
	cin >> userInput;
	cout << endl;

	while (userInput != "stop")
	{
		if (userInput == "RELU")
		{
			TestReLU();
		}

		if (userInput == "MAX")
		{
			TestMaxPool();
		}

		if (userInput == "CONV")
		{
			TestConvolution();
		}

		if (userInput == "TCONV")
		{
			TestTransposeConvolution();
		}

		if (userInput == "SQUISH")
		{
			SquishTest();
		}

		if (userInput == "ERR")
		{
			TestErrorBlock();
		}


		if (userInput == "SUMBLOCK")
		{
			TestSumBlock();
		}


		if (userInput == "ALL")
		{
			TestReLU();
			TestMaxPool();
			TestConvolution();
			TestTransposeConvolution();
			SquishTest();
			TestSumBlock();
			TestErrorBlock();
		}

		cout << "What is your next command?" << endl;
		cin >> userInput;
		cout << endl;
	}

	cout << ">>>> Exiting test mode..." << endl;
}

void NetworkValidator::TestReLU()
{
	CunetaLogger loggy = CunetaLogger();

	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 4;
	int Number_Of_OUTPUT_Layers = 4;

	float** Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	float** Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];

	int max = 2;
	int min = -2;
	int range = max - min + 1;


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = rand() % range + min;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 6;
		}
	}

	ReLU testSubject = ReLU(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject.LayerForwardPass(Forward_Inputs);
	testSubject.LayerBackwardPass(Backprop_Inputs);

	cout << "=====================================================================================================" << endl;
	cout << "============================================ RELU TEST 1 ============================================" << endl;
	cout << "=====================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 4;


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = rand() % range + min;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 6;
		}
	}

	ReLU testSubject2 = ReLU(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject2.LayerForwardPass(Forward_Inputs);
	testSubject2.LayerBackwardPass(Backprop_Inputs);

	cout << "=====================================================================================================" << endl;
	cout << "============================================ RELU TEST 2 ============================================" << endl;
	cout << "=====================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();



	/// <summary>
	/// SWAP NUMBER OF OUTPUT AND INPUT LAYERS
	/// </summary>


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		delete[] Forward_Inputs[j];
	}
	delete[] Forward_Inputs;

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		delete[] Backprop_Inputs[j];
	}
	delete[] Backprop_Inputs;




	Input_HEIGHT = 8;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 4;

	Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];
		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = rand() % range + min;
		}
	}

	Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];
	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];
		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 6;
		}
	}

	ReLU testSubject3 = ReLU(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject3.LayerForwardPass(Forward_Inputs);
	testSubject3.LayerBackwardPass(Backprop_Inputs);

	cout << "=====================================================================================================" << endl;
	cout << "============================================ RELU TEST 3 ============================================" << endl;
	cout << "=====================================================================================================" << endl;
	cout << endl;

	testSubject3.DebugPrintAll();
}

void NetworkValidator::SquishTest()
{
	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 4;
	int Number_Of_OUTPUT_Layers = 1;
	int Filter_Size = 1;
	int Padding = 0;

	float HyperParam_Beta1 = 0.9;
	float HyperParam_Beta2 = 0.9999;
	float HyperParam_Alpha = 0.1;
	float HyperParam_T = 1;
	float HyperParam_Eps = 0.0001;

	float** Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	float** Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = j + 1;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 2;
		}
	}

	Squishy testSubject = Squishy(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);
	testSubject.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject.LayerForwardPass(Forward_Inputs);
	testSubject.LayerBackwardPass(Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ SQUISHY TEST 1 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 1;


	Squishy testSubject2 = Squishy(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);
	testSubject2.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject2.LayerForwardPass(Forward_Inputs);
	testSubject2.LayerBackwardPass(Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ SQUISHY TEST 2 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();
}

void NetworkValidator::TestMaxPool()
{
	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 4;
	int Number_Of_OUTPUT_Layers = 4;

	float** Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	float** Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];

	int max = 2;
	int min = -2;
	int range = max - min + 1;


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = rand() % range + min;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 6;
		}
	}

	MaxPool testSubject = MaxPool(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject.LayerForwardPass(Forward_Inputs);
	testSubject.LayerBackwardPass(Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ MAXPOOL TEST 1 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 4;


	MaxPool testSubject2 = MaxPool(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject2.LayerForwardPass(Forward_Inputs);
	testSubject2.LayerBackwardPass(Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ MAXPOOL TEST 2 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();



	/// <summary>
	/// SWAP NUMBER OF OUTPUT AND INPUT LAYERS
	/// </summary>


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		delete[] Forward_Inputs[j];
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		delete[] Backprop_Inputs[j];
	}


	delete[] Backprop_Inputs;
	delete[] Forward_Inputs;




	Input_HEIGHT = 8;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 4;

	Forward_Inputs = new float* [Number_Of_INPUT_Layers];

	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];
		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = rand() % range + min;
		}
	}

	float** New_Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		New_Backprop_Inputs[j] = new float[Input_LENGTH];
		for (int i = 0; i < Input_LENGTH; i++)
		{
			New_Backprop_Inputs[j][i] = 6;
		}
	}

	MaxPool testSubject3 = MaxPool(Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH, 0, 0);

	testSubject3.LayerForwardPass(Forward_Inputs);
	testSubject3.LayerBackwardPass(New_Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ MAXPOOL TEST 3 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject3.DebugPrintAll();
}

void NetworkValidator::TestConvolution()
{
	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 4;
	int Number_Of_OUTPUT_Layers = 2;
	int Filter_Size = 3;
	int Padding = 2;

	float HyperParam_Beta1 = 0.9;
	float HyperParam_Beta2 = 0.9999;
	float HyperParam_Alpha = 0.1;
	float HyperParam_T = 1;
	float HyperParam_Eps = 0.0001;

	float** Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	float** Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = j + 1;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 2;
		}
	}

	Convolution testSubject = Convolution(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH);
	testSubject.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject.LayerForwardPass(Forward_Inputs);
	testSubject.LayerBackwardPass(Backprop_Inputs);

	cout << "================================================================================================================================" << endl;
	cout << "====================================================== CONVOLUTION TEST 1 ======================================================" << endl;
	cout << "================================================================================================================================" << endl;
	cout << endl;

	//testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 2;


	Convolution testSubject2 = Convolution(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH);
	testSubject2.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject2.LayerForwardPass(Forward_Inputs);
	testSubject2.LayerBackwardPass(Backprop_Inputs);

	cout << "================================================================================================================================" << endl;
	cout << "====================================================== CONVOLUTION TEST 2 ======================================================" << endl;
	cout << "================================================================================================================================" << endl;
	cout << endl;

	//testSubject2.DebugPrintAll();
}

void NetworkValidator::TestErrorBlock()
{
	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;


	float* Forward_Inputs = new float[Input_LENGTH];
	float* Ground_Truth_Input = new float[Input_LENGTH];

	int max = 1;
	int min = 0;
	int range = max - min + 1;


	for (int i = 0; i < Input_LENGTH; i++)
	{
		Forward_Inputs[i] = rand() % range + min;
	}

	for (int i = 0; i < Input_LENGTH; i++)
	{
		Ground_Truth_Input[i] = rand() % range + min;
	}


	ErrorCalcModule testSubject = ErrorCalcModule(Forward_Inputs, Ground_Truth_Input, Input_HEIGHT, Input_WIDTH);
	testSubject.LayerForwardPass();
	cout << "========================================================================================================" << endl;
	cout << "============================================ SumBlock TEST 1 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;

	ErrorCalcModule testSubject2 = ErrorCalcModule(Forward_Inputs, Ground_Truth_Input, Input_HEIGHT, Input_WIDTH);
	testSubject2.LayerForwardPass();
	cout << "========================================================================================================" << endl;
	cout << "============================================ SumBlock TEST 2 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();
}

void NetworkValidator::TestSumBlock()
{
	int Input_HEIGHT = 8;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 4;


	float** InputSet_1 = new float* [Number_Of_INPUT_Layers];
	float** InputSet_2 = new float* [Number_Of_INPUT_Layers];


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		InputSet_1[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			InputSet_1[j][i] = 1;
		}
	}

	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		InputSet_2[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			InputSet_2[j][i] = 2;
		}
	}

	SumBlock testSubject = SumBlock(Input_HEIGHT, Input_WIDTH, Number_Of_INPUT_Layers, 0, 0);
	testSubject.Sum(InputSet_1, InputSet_2);

	cout << "========================================================================================================" << endl;
	cout << "============================================ SumBlock TEST 1 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;


	SumBlock testSubject2 = SumBlock(Input_HEIGHT, Input_WIDTH, Number_Of_INPUT_Layers, 0, 0);
	testSubject2.Sum(InputSet_1, InputSet_2);

	cout << "========================================================================================================" << endl;
	cout << "============================================ SumBlock TEST 2 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();
}

void NetworkValidator::TestTransposeConvolution()
{
	int Input_HEIGHT = 6;
	int Input_WIDTH = 4;
	int Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	int Number_Of_INPUT_Layers = 2;
	int Number_Of_OUTPUT_Layers = 4;
	int Filter_Size = 3;
	int Padding = 2;

	float HyperParam_Beta1 = 0.9;
	float HyperParam_Beta2 = 0.9999;
	float HyperParam_Alpha = 0.1;
	float HyperParam_T = 1;
	float HyperParam_Eps = 0.0001;

	float** Forward_Inputs = new float* [Number_Of_INPUT_Layers];
	float** Backprop_Inputs = new float* [Number_Of_OUTPUT_Layers];


	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		Forward_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Forward_Inputs[j][i] = j + 1;
		}
	}

	for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	{
		Backprop_Inputs[j] = new float[Input_LENGTH];

		for (int i = 0; i < Input_LENGTH; i++)
		{
			Backprop_Inputs[j][i] = 2;
		}
	}

	TransposeConvolution testSubject = TransposeConvolution(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH);
	testSubject.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject.LayerForwardPass(Forward_Inputs);
	testSubject.LayerBackwardPass(Backprop_Inputs);

	cout << "================================================================================================================================" << endl;
	cout << "====================================================== CONVOLUTION TEST 1 ======================================================" << endl;
	cout << "================================================================================================================================" << endl;
	cout << endl;

	testSubject.DebugPrintAll();

	/// <summary>
	/// SWAP WIDTH AND HEIGHT DIMENSIONS
	/// </summary>

	Input_HEIGHT = 4;
	Input_WIDTH = 6;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 2;
	Number_Of_OUTPUT_Layers = 4;


	TransposeConvolution testSubject2 = TransposeConvolution(Filter_Size, Padding, Number_Of_INPUT_Layers, Number_Of_OUTPUT_Layers, Input_HEIGHT, Input_WIDTH);
	testSubject2.SetHyperParams(HyperParam_Beta1, HyperParam_Beta2, HyperParam_Eps, HyperParam_T, HyperParam_Alpha);
	testSubject2.LayerForwardPass(Forward_Inputs);
	testSubject2.LayerBackwardPass(Backprop_Inputs);

	cout << "================================================================================================================================" << endl;
	cout << "====================================================== CONVOLUTION TEST 2 ======================================================" << endl;
	cout << "================================================================================================================================" << endl;
	cout << endl;

	testSubject2.DebugPrintAll();
}








