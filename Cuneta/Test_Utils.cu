#include "Test_Utils.cuh"

#include <iostream>

#include "Convolution.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
using namespace std;

void TestReLU(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight)
{
	cout << "Starting ReLU Test" << endl;
	cout << "Original matrix:" << endl;
	int rowCounter = 0;
	for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << inputMatrix[i] << " ";
		rowCounter++;
		if (rowCounter == 15)
		{
			cout << endl;
			rowCounter = 0;
		}
	}
	cout << endl;

	ReLU testSubject = ReLU(inputMatrix, outputMatrix, matrixHeight, matrixHeight, matrixWidth, matrixWidth);

	cout << "Starting forward pass test." << endl;
	testSubject.ForwardPass();

	cout << "Matrix after ReLU:" << endl;
	rowCounter = 0;

	testSubject.m_OutputMatrix[2];

	for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << testSubject.m_OutputMatrix[i] << " ";
		rowCounter++;
		if (rowCounter == 15)
		{
			cout << endl;
			rowCounter = 0;
		}
	}
	cout << endl;
}

void TestMaxPool(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight)
{
	cout << "Starting MaxPool Test" << endl;
	cout << "Original matrix:" << endl;
	for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << inputMatrix[i] << " ";
	}
	cout << endl;

	MaxPool testSubject = MaxPool(inputMatrix, matrixHeight, matrixHeight);

	cout << "Starting forward pass test." << endl;
	testSubject.ForwardPass();

	cout << "Matrix after MaxPool:" << endl;


	for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
	{
		cout << testSubject.m_OutputMatrix[i] << " ";
	}
	cout << endl;
}


void TestConvolution(float* inputMatrix, int matrixWidth, int matrixHeight, int filterSize)
{
	cout << "Starting MaxPool Test" << endl;
	cout << "Original matrix:" << endl;
	for (int i = 0; i < matrixWidth * matrixHeight; ++i)
	{
		cout << inputMatrix[i] << " ";
	}
	cout << endl;

	Convolution testSubject = Convolution(inputMatrix, matrixHeight, matrixWidth, filterSize);

	cout << "Generated filter:" << endl;
	testSubject.ForwardPass();

	for (int i = 0; i < testSubject.filterSize * testSubject.filterSize; ++i)
	{
		cout << testSubject.filter[i] << " ";
	}

	cout << endl;


	cout << "Generated Toeplitz matrix:" << endl;
	int numberOfInputElements = testSubject.m_InputMatrixHeight * testSubject.m_InputMatrixWidth;
	int numberOfOutputElements = testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth;
	cout << "Toeplitz matrix element count: " << numberOfInputElements * numberOfOutputElements << endl;

	for (int i = 0; i < numberOfInputElements * numberOfOutputElements; ++i)
	{
		cout << testSubject.toeplitzMatrix[i] << " ";
		if (i % 20 == 0)
		{
			cout << endl;
		}
	}
	cout << endl;
}

