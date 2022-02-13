#include "Test_Utils.cuh"

#include <iostream>


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


	/*for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
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
	}*/

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


	//for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	//{
	//	delete[] Forward_Inputs[j];
	//}

	//for (int j = 0; j < Number_Of_OUTPUT_Layers; ++j)
	//{
	//	delete[] Backprop_Inputs[j];
	//}


	//delete[] Backprop_Inputs;
	//delete[] Forward_Inputs;




	Input_HEIGHT = 8;
	Input_WIDTH = 8;
	Input_LENGTH = Input_HEIGHT * Input_WIDTH;
	Number_Of_INPUT_Layers = 4;
	Number_Of_OUTPUT_Layers = 4;

	float** New_Forward_Inputs = new float* [Number_Of_INPUT_Layers];

	for (int j = 0; j < Number_Of_INPUT_Layers; ++j)
	{
		New_Forward_Inputs[j] = new float[Input_LENGTH];
		for (int i = 0; i < Input_LENGTH; i++)
		{
			New_Forward_Inputs[j][i] = rand() % range + min;
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

	testSubject3.LayerForwardPass(New_Forward_Inputs);
	testSubject3.LayerBackwardPass(New_Backprop_Inputs);

	cout << "========================================================================================================" << endl;
	cout << "============================================ MAXPOOL TEST 3 ============================================" << endl;
	cout << "========================================================================================================" << endl;
	cout << endl;

	testSubject3.DebugPrintAll();
}

void NetworkValidator::TestConvolution()
{

}

void NetworkValidator::TestErrorBlock()
{

}

void NetworkValidator::TestSumBlock()
{

}

void NetworkValidator::TestTransposeConvolution()
{

}








