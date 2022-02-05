#include "Test_Utils.cuh"

#include <iostream>

#include "Convolution.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "TransposeConvolution.cuh"
using namespace std;

void TestReLU(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight)
{
	cout << "Starting ReLU Test" << endl;
	cout << "Original matrix:" << endl;
	int rowCounter = 0;
	/*for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << inputMatrix[i] << " ";
		rowCounter++;
		if (rowCounter == 15)
		{
			cout << endl;
			rowCounter = 0;
		}
	}
	cout << endl;*/

	ReLU testSubject = ReLU(inputMatrix, outputMatrix, matrixHeight, matrixHeight, matrixWidth, matrixWidth);

	cout << "Starting forward pass test." << endl;
	testSubject.ForwardPass();

	cout << "Matrix after ReLU:" << endl;
	rowCounter = 0;

	testSubject.m_OutputMatrix[2];

	/*for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << testSubject.m_OutputMatrix[i] << " ";
		rowCounter++;
		if (rowCounter == 15)
		{
			cout << endl;
			rowCounter = 0;
		}
	}
	cout << endl;*/
}

void TestMaxPool(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight, bool printMatricies)
{
	cout << "Starting MaxPool Test" << endl;
	cout << "Original matrix:" << endl;
	int counter = 1;
	if (printMatricies) {

		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
		{
			cout << inputMatrix[i] << " ";
			counter++;
			if (counter == matrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}

		}
	}
	cout << endl;

	MaxPool testSubject = MaxPool(inputMatrix, matrixHeight, matrixWidth);

	cout << "Starting forward pass test." << endl;
	testSubject.ForwardPass();

	cout << "Matrix after MaxPool:" << endl;

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
	cout << testSubject.m_OutputMatrix[0] << " ";
}


void TestConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies)
{
	cout << "Starting Convolution Test" << endl;
	cout << "Original matrix:" << endl;

	int counter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
		{
			cout << inputMatrix[i] << " ";

			counter++;

			if (counter == matrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;

	}

	Convolution testSubject = Convolution(inputMatrix, matrixHeight, matrixWidth, filterSize);

	cout << "Generated filter:" << endl;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.filterSize * testSubject.filterSize; ++i)
		{
			cout << testSubject.filter[i] << " ";

			counter++;

			if (counter == testSubject.filterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}

		cout << endl;
	}

	cout << "Starting forward pass test" << endl;

	testSubject.ForwardPass();

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
		cout << endl;
	}
	cout << "First item in convolution result: " << testSubject.m_OutputMatrix[0] << endl;
}


void TestTransposeConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies)
{
	cout << "Starting Transpose Convolution Test" << endl;
	cout << "Original matrix:" << endl;

	int counter = 1;

	if (printMatricies)
	{
		for (int i = 0; i < matrixWidth * matrixHeight; ++i)
		{
			cout << inputMatrix[i] << " ";

			counter++;

			if (counter == matrixWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;

	}

	TransposeConvolution testSubject = TransposeConvolution(inputMatrix, matrixHeight, matrixWidth, filterSize, 2);

	cout << "Generated filter:" << endl;

	if (printMatricies)
	{
		for (int i = 0; i < testSubject.filterSize * testSubject.filterSize; ++i)
		{
			cout << testSubject.filter[i] << " ";

			counter++;

			if (counter == testSubject.filterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}

		cout << endl;
	}

	cout << "Result from padding input" << endl;
	if (printMatricies)
	{
		for (int i = 0; i < testSubject.paddedInputHeight * testSubject.paddedInputWidth; ++i)
		{
			cout << testSubject.paddedInput[i] << " ";

			counter++;

			if (counter == testSubject.paddedInputWidth + 1)
			{
				cout << endl;
				counter = 1;
			}
		}

		cout << endl;
	}

	cout << "Starting forward pass test" << endl;

	testSubject.ForwardPass();

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
		cout << endl;
	}
	cout << "First item in convolution result: " << testSubject.m_OutputMatrix[0] << endl;
}
