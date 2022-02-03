#include "Test_Utils.cuh"

#include <iostream>

#include "MaxPool.cuh"
#include "ReLU.cuh"
using namespace std;

void TestReLU(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight)
{
	cout << "Starting ReLU Test" << endl;
	cout << "Original matrix:" << endl;
	int rowCounter = 0;
	for (int i = 0; i < matrixWidth*matrixHeight; ++i)
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

	cout << "Matrix after ReLU:" << endl;


	for (int i = 0; i < testSubject.m_OutputMatrixHeight * testSubject.m_OutputMatrixWidth; ++i)
	{
		cout << testSubject.m_OutputMatrix[i] << " ";
	}
	cout << endl;
}


