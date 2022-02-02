#include "ReLU_Test.cuh"

#include <iostream>
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
