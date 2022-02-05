#include "ImageIngester.cuh";

#include <fstream>
#include <iostream>

using namespace std;

ImageIngester::ImageIngester(string _inputPath, string _groundTruthPath)
{
	inputPath = _inputPath;
	groundTruthPath = _groundTruthPath;
}


void ImageIngester::ReadData()
{

	ifstream inFile;
	inFile.open(inputPath);

	if (inFile.is_open())
	{
		inFile >> inputHeight;
		inFile >> inputWidth;

		inputImageData = new float[inputHeight * inputWidth];
		int inputSize = 0;
		for (int i = 0; i < inputHeight * inputWidth; i++)
		{
			inFile >> inputImageData[i];
			inputSize++;
		}

		inFile.close(); // CLose input file
		cout << "Input height: " << inputHeight << " " << "Input width: " << inputWidth << endl;
		cout << "Input size: " << inputSize << endl;
	}
	else { //Error message
		cerr << "Can't find input file " << inputPath << endl;
	}

	inFile.open(groundTruthPath);

	if (inFile.is_open())
	{
		inFile >> groundTruthHeight;
		inFile >> groundTruthWidth;

		groundTruthData = new float[groundTruthHeight * groundTruthWidth];
		int inputSize = 0;
		for (int i = 0; i < groundTruthHeight * groundTruthWidth; i++)
		{
			inFile >> groundTruthData[i];
			inputSize++;
		}

		inFile.close(); // CLose input file
		cout << "Input height: " << groundTruthHeight << " " << "Input width: " << groundTruthWidth << endl;
		cout << "Input size: " << inputSize << endl;
	}
	else { //Error message
		cerr << "Can't find input file " << groundTruthPath << endl;
	}
}


