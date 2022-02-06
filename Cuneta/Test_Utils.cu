#include "Test_Utils.cuh"

#include <iostream>

#include "Convolution.cuh"
#include "ErrorCalcModule.cuh"
#include "ImageIngester.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "TransposeConvolution.cuh"
using namespace std;

void TestReLU(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies)
{
	cout << "Starting ReLU Test" << endl;

	size_t inputVectorizedSize = (size_t)inputMatrixWidth * (size_t)inputMatrixHeight;
	float* input = new float[inputVectorizedSize];


	int max = maxInputVal;
	int min = minInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; i++)
	{
		input[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int rowCounter = 1;
	cout << "Original matrix:" << endl;
	if (printMatricies)
	{
		for (size_t i = 0; i < inputMatrixWidth * inputMatrixHeight; i++)
		{
			float x = input[i];
			cout << input[i] << " ";
			rowCounter++;
			if (rowCounter == inputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	ReLU testSubject = ReLU();

	cout << "Starting forward pass." << endl;
	testSubject.ForwardPass(input, inputMatrixHeight, inputMatrixWidth);

	cout << "ReLU output:" << endl;

	cout << "Output dimensions (Height/Width): " << testSubject.m_OutputMatrixHeight << "/" << testSubject.m_OutputMatrixWidth << endl;

	cout << "Output matrix:" << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixWidth * inputMatrixHeight; ++i)
		{
			cout << testSubject.m_OutputMatrix[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_OutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	cout << "Forward ReLU test complete!" << endl << endl;

	delete[] input;
}

void TestBackpropReLU(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies)
{
	cout << "Starting Backprop ReLU Test" << endl;

	int inputVectorizedSize = inputMatrixHeight * inputMatrixWidth;
	float* fwdInput = new float[inputVectorizedSize];
	float* backInput = new float[inputVectorizedSize];

	//Initialize dummy forward input;
	int min = fwdMinInputVal;
	int max = fwdMaxInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; ++i)
	{
		fwdInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	//Initialize backprop input;
	min = backMinInputVal;
	max = backMaxInputVal;
	range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; ++i)
	{
		backInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int rowCounter = 1;
	cout << "Forward input matrix:" << endl;
	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixWidth * inputMatrixHeight; ++i)
		{
			cout << fwdInput[i] << " ";
			rowCounter++;
			if (rowCounter == inputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	cout << "Backprop input matrix:" << endl;

	rowCounter = 1;
	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixWidth * inputMatrixHeight; ++i)
		{
			cout << backInput[i] << " ";
			rowCounter++;
			if (rowCounter == inputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}


	ReLU testSubject = ReLU();

	cout << "Starting backward pass." << endl;
	testSubject.ForwardPass(fwdInput, inputMatrixHeight, inputMatrixWidth);

	cout << "Starting backward pass." << endl;
	testSubject.BackwardPass(backInput, inputMatrixHeight, inputMatrixWidth);

	cout << "ReLU backprop output:" << endl;

	cout << "Backprop Output dimensions (Height/Width): " << testSubject.m_BackpropOutputMatrixHeight << "/" << testSubject.m_BackpropOutputMatrixWidth << endl;

	cout << "Backprop Output matrix:" << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixWidth * inputMatrixHeight; ++i)
		{
			cout << testSubject.m_BackpropagationOutput[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_BackpropOutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	cout << "Backpropagation ReLU test complete!" << endl << endl;
	delete[] fwdInput;
	delete[] backInput;
}


void TestMaxPool(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies)
{
	cout << "Starting MaxPool Test" << endl;

	size_t inputVectorizedSize = (size_t)inputMatrixWidth * (size_t)inputMatrixHeight;
	float* input = new float[inputVectorizedSize];


	int max = maxInputVal;
	int min = minInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; i++)
	{
		input[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int counter = 1;
	cout << "Original matrix:" << endl;
	if (printMatricies)
	{
		for (size_t i = 0; i < inputMatrixWidth * inputMatrixHeight; i++)
		{
			float x = input[i];
			cout << input[i] << " ";
			counter++;
			if (counter == inputMatrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}

	MaxPool testSubject = MaxPool();

	cout << "Starting forward pass test." << endl;
	testSubject.ForwardPass(input, inputMatrixHeight, inputMatrixWidth);

	cout << "Output dimensions (Height/Width): " << testSubject.m_OutputMatrixHeight << "/" << testSubject.m_OutputMatrixWidth << endl;

	cout << "Output matrix:" << endl;

	counter = 1;
	if (printMatricies) {
		for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
		{
			cout << testSubject.m_OutputMatrix[i] << " ";
			counter++;

			if (counter == testSubject.m_OutputMatrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
	}
	cout << endl;
	cout << "Forward MaxPool test complete!" << endl << endl;


	delete[] input;
}

void TestBackpropMaxPool(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies)
{
	cout << "Starting Backprop MaxPool Test" << endl;

	int inputVectorizedSize = inputMatrixHeight * inputMatrixWidth;
	int backpropInputVectorizedSize = (inputMatrixHeight / 2) * (inputMatrixWidth / 2);
	float* fwdInput = new float[inputVectorizedSize];
	float* backInput = new float[backpropInputVectorizedSize];

	//Initialize dummy forward input;
	int min = fwdMinInputVal;
	int max = fwdMaxInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; ++i)
	{
		fwdInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	//Initialize backprop input;
	min = backMinInputVal;
	max = backMaxInputVal;
	range = max - min + 1;

	for (int i = 0; i < backpropInputVectorizedSize; ++i)
	{
		backInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int rowCounter = 1;
	cout << "Forward input matrix:" << endl;
	if (printMatricies)
	{
		for (int i = 0; i < inputVectorizedSize; ++i)
		{
			cout << fwdInput[i] << " ";
			rowCounter++;
			if (rowCounter == inputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	cout << "Backprop input matrix:" << endl;

	rowCounter = 1;
	if (printMatricies)
	{
		for (int i = 0; i < backpropInputVectorizedSize; ++i)
		{
			cout << backInput[i] << " ";
			rowCounter++;
			if (rowCounter == (inputMatrixWidth / 2) + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}


	MaxPool testSubject = MaxPool();

	cout << "Starting forward pass." << endl;
	testSubject.ForwardPass(fwdInput, inputMatrixHeight, inputMatrixWidth);

	cout << "Forward pass output:" << endl;
	cout << "Output dimensions (Height/Width): " << testSubject.m_OutputMatrixHeight << "/" << testSubject.m_OutputMatrixWidth << endl;

	cout << "Output matrix:" << endl;

	rowCounter = 1;
	if (printMatricies) {
		for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
		{
			cout << testSubject.m_OutputMatrix[i] << " ";
			rowCounter++;

			if (rowCounter == testSubject.m_OutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}
	cout << endl;


	cout << "Starting backward pass." << endl;
	testSubject.BackwardPass(backInput, testSubject.m_OutputMatrixHeight, testSubject.m_OutputMatrixWidth);

	cout << "MaxPool backprop output:" << endl;

	cout << "Backprop Output dimensions (Height/Width): " << testSubject.m_BackpropOutputMatrixHeight << "/" << testSubject.m_BackpropOutputMatrixWidth << endl;

	cout << "Backprop Output matrix:" << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixWidth * inputMatrixHeight; ++i)
		{
			cout << testSubject.m_BackpropagationOutput[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_BackpropOutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}

	cout << "Backpropagation MaxPool test complete!" << endl << endl;
	delete[] fwdInput;
	delete[] backInput;
}



void TestConvolution(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, int filterSize, int paddingSize, bool printMatricies)
{
	cout << "Starting Convolution Test" << endl;
	cout << "Original matrix:" << endl;
	cout << endl;

	size_t inputVectorizedSize = (size_t)inputMatrixWidth * (size_t)inputMatrixHeight;
	float* input = new float[inputVectorizedSize];


	int max = maxInputVal;
	int min = minInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; i++)
	{
		input[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int counter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; i++)
		{
			cout << input[i] << " ";

			counter++;

			if (counter == inputMatrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	Convolution testSubject = Convolution(filterSize, paddingSize);

	cout << "Generated filter:" << endl;
	cout << endl;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_FilterSize * testSubject.m_FilterSize; ++i)
		{
			cout << testSubject.m_Filter[i] << " ";

			counter++;

			if (counter == testSubject.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}


	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;


	cout << "Starting forward pass test" << endl;

	testSubject.ForwardPass(input, inputMatrixHeight, inputMatrixWidth);

	cout << "Output dimensions (Height/Width): " << testSubject.m_OutputMatrixHeight << "/" << testSubject.m_OutputMatrixWidth << endl;
	cout << "Output matrix:" << endl;
	cout << endl;
	counter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
		{
			cout << testSubject.m_OutputMatrix[i] << " ";

			counter++;

			if (counter == testSubject.m_OutputMatrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << "Convolution test complete!" << endl << endl;
	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << endl << ">>>>>>>>>>>>>>>> NEXT TEST >>>>>>>>>>>>>>>>";
	cout << endl << ">>>>>>>>>>>>>>>> NEXT TEST >>>>>>>>>>>>>>>>";
	delete[] input;
}

void TestBackpropConvolution(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, int filterSize, int paddingSize, bool printMatricies)
{
	cout << endl;
	cout << endl;
	cout << "Starting Backprop MaxPool Test" << endl;

	int inputVectorizedSize = inputMatrixHeight * inputMatrixWidth;
	int backpropInputVectorizedSize = (inputMatrixHeight - 2) * (inputMatrixWidth - 2);
	float* fwdInput = new float[inputVectorizedSize];
	float* backInput = new float[backpropInputVectorizedSize];

	//Initialize dummy forward input;
	int min = fwdMinInputVal;
	int max = fwdMaxInputVal;
	int range = max - min + 1;

	for (int i = 0; i < inputMatrixHeight * inputMatrixWidth; ++i)
	{
		fwdInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	//Initialize backprop input;
	min = backMinInputVal;
	max = backMaxInputVal;
	range = max - min + 1;

	for (int i = 0; i < backpropInputVectorizedSize; ++i)
	{
		backInput[i] = rand() % range + min;
		//std::cout << input[i] << std::endl;
	}

	int rowCounter = 1;
	cout << "Forward input matrix:" << endl;
	cout << endl;
	if (printMatricies)
	{
		for (int i = 0; i < inputVectorizedSize; ++i)
		{
			cout << fwdInput[i] << " ";
			rowCounter++;
			if (rowCounter == inputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;
	
	cout << "Backprop input matrix:" << endl;
	cout << endl;
	rowCounter = 1;
	if (printMatricies)
	{
		for (int i = 0; i < backpropInputVectorizedSize; ++i)
		{
			cout << backInput[i] << " ";
			rowCounter++;
			if (rowCounter == (inputMatrixWidth - 2) + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}
	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	Convolution testSubject = Convolution(filterSize, paddingSize);

	cout << "Starting forward pass." << endl;
	testSubject.ForwardPass(fwdInput, inputMatrixHeight, inputMatrixWidth);

	cout << "Forward pass output:" << endl;
	cout << "Output dimensions (Height/Width): " << testSubject.m_OutputMatrixHeight << "/" << testSubject.m_OutputMatrixWidth << endl;

	cout << "Output matrix:" << endl;
	cout << endl;

	rowCounter = 1;
	if (printMatricies) {
		for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
		{
			cout << testSubject.m_OutputMatrix[i] << " ";
			rowCounter++;

			if (rowCounter == testSubject.m_OutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;


	cout << "Starting backward pass." << endl;
	testSubject.BackwardPass(backInput, testSubject.m_OutputMatrixHeight, testSubject.m_OutputMatrixWidth);

	cout << "Convolution backprop output:" << endl;
	cout << "Filter Backprop Output dimensions (Height/Width): " << testSubject.m_BackpropOutputMatrixHeight << "/" << testSubject.m_BackpropOutputMatrixWidth << endl;
	cout << "Filter Backprop Output matrix:" << endl;
	cout << endl;
	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_BackpropOutputMatrixHeight * testSubject.m_BackpropOutputMatrixWidth; ++i)
		{
			cout << testSubject.m_BackpropagationOutput[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_BackpropOutputMatrixWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << "Padded fwd output:" << endl;
	cout << "Padded Output dimensions (Height/Width): " << testSubject.m_PaddedInputHeight << "/" << testSubject.m_PaddedInputWidth<< endl;
	cout << "Padded matrix:" << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_PaddedInputHeight * testSubject.m_PaddedInputWidth; ++i)
		{
			cout << testSubject.m_PaddedBackpropInput[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_PaddedInputWidth + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}


	cout << endl << "=====================================";
	cout << endl;
	cout << endl;


	cout << "Filter output:" << endl;
	cout << "Filter dimensions (Height/Width): " << testSubject.m_FilterSize << "/" << testSubject.m_FilterSize << endl;
	cout << "Filter matrix:" << endl;
	cout << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_FilterSize * testSubject.m_FilterSize; ++i)
		{
			cout << testSubject.m_Filter[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_FilterSize + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << "Flipped filter output:" << endl;
	cout << "Padded Output dimensions (Height/Width): " << testSubject.m_FilterSize << "/" << testSubject.m_FilterSize << endl;
	cout << "Flipped filter matrix:" << endl;
	cout << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_FilterSize * testSubject.m_FilterSize; ++i)
		{
			cout << testSubject.m_FlippedFilter[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_FilterSize + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
	}

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << "Filter backprop output:" << endl;
	cout << "Backprop Output dimensions (Height/Width): " << testSubject.m_FilterSize << "/" << testSubject.m_FilterSize << endl;
	cout << "Backprop Output matrix:" << endl;
	cout << endl;

	rowCounter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.m_FilterSize * testSubject.m_FilterSize; ++i)
		{
			cout << testSubject.m_FilterBackpropResult[i] << " ";
			rowCounter++;
			if (rowCounter == testSubject.m_FilterSize + 1)
			{
				cout << endl;
				rowCounter = 1;
			}
		}
		cout << endl;
	}
	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << "Backpropagation convolution test complete!" << endl << endl;

	cout << endl << "=====================================";
	cout << endl;
	cout << endl;

	cout << endl << ">>>>>>>>>>>>>>>> NEXT TEST >>>>>>>>>>>>>>>>";
	cout << endl << ">>>>>>>>>>>>>>>> NEXT TEST >>>>>>>>>>>>>>>>";
	
	
	delete[] fwdInput;
	delete[] backInput;
}


//
//void TestTransposeConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies)
//{
//	cout << "Starting Transpose Convolution Test" << endl;
//	cout << "Original matrix:" << endl;
//
//	int counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
//		{
//			cout << inputMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == matrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//		cout << endl;
//
//	}
//
//	TransposeConvolution testSubject = TransposeConvolution(inputMatrix, matrixHeight, matrixWidth, filterSize, 2);
//
//	cout << "Generated m_Filter:" << endl;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_FilterSize * testSubject.m_FilterSize; ++i)
//		{
//			cout << testSubject.m_Filter[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.m_FilterSize + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//
//		cout << endl;
//	}
//
//	cout << "Result from padding input" << endl;
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_PaddedInputHeight * testSubject.m_PaddedInputWidth; ++i)
//		{
//			cout << testSubject.m_PaddedInput[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.m_PaddedInputWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//
//		cout << endl;
//	}
//
//	cout << "Starting forward pass test" << endl;
//
//	testSubject.ForwardPass();
//
//	counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
//		{
//			cout << testSubject.m_OutputMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.m_OutputMatrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//		cout << endl;
//	}
//	cout << "First item in convolution result: " << testSubject.m_OutputMatrix[0] << endl;
//}
//
//
//void TestErrorCalcModule(float* inputMatrix, float* groundTruthMatrix, int matrixHeight, int matrixWidth, bool printMatricies)
//{
//	cout << "Starting Error Calc Test" << endl;
//	cout << "Original matrix:" << endl;
//
//	int counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
//		{
//			cout << inputMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == matrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//		cout << endl;
//	}
//
//	cout << "Ground truth matrix:" << endl;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
//		{
//			cout << groundTruthMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == matrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//		cout << endl;
//
//	}
//
//	ErrorCalcModule testSubject = ErrorCalcModule(inputMatrix, groundTruthMatrix, matrixHeight, matrixWidth);
//
//	testSubject.PixelWiseSigmoid();
//
//	cout << "Sigmoid result:" << endl;
//	counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_InputMatrixWidth * testSubject.m_InputMatrixHeight; ++i)
//		{
//			cout << testSubject.sigmoidResultMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.m_InputMatrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//
//		cout << endl;
//	}
//
//	testSubject.PixelWiseCrossEntropy();
//	cout << "Cross entropy result: " << endl;
//	counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_InputMatrixWidth * testSubject.m_InputMatrixHeight; ++i)
//		{
//			cout << testSubject.crossEntropyResultMatrix[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.m_InputMatrixWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//
//		cout << endl;
//	}
//
//	cout << "Network error: " << endl;
//
//	testSubject.CrossEntropySum();
//
//	counter = 1;
//
//	if (printMatricies)
//	{
//		for (int i = 0; i < testSubject.m_OutputMatrixHeight; ++i)
//		{
//			cout << testSubject.intermediateSumResult[i] << " ";
//		}
//		cout << endl;
//	}
//	cout << "Error calc module test complete." << endl;
//}
//
//
//void TestImageIngester(string inputPath, string groundTruthPath, bool printMatricies)
//{
//	ImageIngester testSubject = ImageIngester(inputPath, groundTruthPath);
//	testSubject.ReadData();
//
//	if (printMatricies)
//	{
//		int counter = 1;
//		for (int i = 0; i < testSubject.inputHeight * testSubject.inputWidth; ++i)
//		{
//			cout << testSubject.inputImageData[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.inputWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//		
//		counter = 1;
//		for (int i = 0; i < testSubject.groundTruthHeight * testSubject.groundTruthWidth; ++i)
//		{
//			cout << testSubject.groundTruthData[i] << " ";
//
//			counter++;
//
//			if (counter == testSubject.groundTruthWidth + 1)
//			{
//				cout << endl;
//				counter = 1;
//			}
//		}
//	}
//}
