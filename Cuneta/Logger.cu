#include "Logger.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
using namespace std;
void CunetaLogger::LogReLUState(ReLU reluToSave, string outputDirectory, string imageName, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "ReLU_";
	logFilePath += imageName;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);


	output << "RELU" << endl;
	output << imageName << endl;
	output << iteration << endl;

	output << endl;
	output << "m_InputDims" << endl;
	output << reluToSave.m_InputMatrixHeight << endl;
	output << reluToSave.m_InputMatrixWidth << endl;

	output << endl;
	output << "m_OutputDims" << endl;
	output << reluToSave.m_OutputMatrixHeight << endl;
	output << reluToSave.m_OutputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropInputDims" << endl;
	output << reluToSave.m_BackpropInputMatrixHeight << endl;
	output << reluToSave.m_BackpropInputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropOutputDims" << endl;
	output << reluToSave.m_BackpropOutputMatrixHeight << endl;
	output << reluToSave.m_BackpropOutputMatrixWidth << endl;

	output << endl;
	output << "m_InputMatrix" << endl;
	for (int i = 0; i < reluToSave.m_InputMatrixHeight * reluToSave.m_InputMatrixWidth; ++i)
	{
		output << reluToSave.m_InputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_OutputMatrix" << endl;

	for (int i = 0; i < reluToSave.m_OutputMatrixHeight * reluToSave.m_OutputMatrixWidth; ++i)
	{
		output << reluToSave.m_OutputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackPropInputMatrix" << endl;

	for (int i = 0; i < reluToSave.m_BackpropInputMatrixHeight * reluToSave.m_BackpropInputMatrixWidth; ++i)
	{
		output << reluToSave.m_BackPropInputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackpropagationOutput" << endl;

	for (int i = 0; i < reluToSave.m_BackpropOutputMatrixHeight * reluToSave.m_BackpropOutputMatrixWidth; ++i)
	{
		output << reluToSave.m_BackpropagationOutput[i] << " ";
	}
	output << endl;
	output.close();
}

void CunetaLogger::LogMaxPoolState(MaxPool maxPoolToSave, string outputDirectory, string imageName, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "MaxPool_";
	logFilePath += imageName;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);


	output << "MAXPOOL" << endl;
	output << imageName << endl;
	output << iteration << endl;

	output << endl;
	output << "m_InputDims" << endl;
	output << maxPoolToSave.m_InputMatrixHeight << endl;
	output << maxPoolToSave.m_InputMatrixWidth << endl;

	output << endl;
	output << "m_OutputDims" << endl;
	output << maxPoolToSave.m_OutputMatrixHeight << endl;
	output << maxPoolToSave.m_OutputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropInputDims" << endl;
	output << maxPoolToSave.m_BackpropInputMatrixHeight << endl;
	output << maxPoolToSave.m_BackpropInputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropOutputDims" << endl;
	output << maxPoolToSave.m_BackpropOutputMatrixHeight << endl;
	output << maxPoolToSave.m_BackpropOutputMatrixWidth << endl;

	output << endl;
	output << "m_InputMatrix" << endl;
	for (int i = 0; i < maxPoolToSave.m_InputMatrixHeight * maxPoolToSave.m_InputMatrixWidth; ++i)
	{
		output << maxPoolToSave.m_InputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_OutputMatrix" << endl;

	for (int i = 0; i < maxPoolToSave.m_OutputMatrixHeight * maxPoolToSave.m_OutputMatrixWidth; ++i)
	{
		output << maxPoolToSave.m_OutputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackPropInputMatrix" << endl;

	for (int i = 0; i < maxPoolToSave.m_BackpropInputMatrixHeight * maxPoolToSave.m_BackpropInputMatrixWidth; ++i)
	{
		output << maxPoolToSave.m_BackPropInputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackpropagationOutput" << endl;

	for (int i = 0; i < maxPoolToSave.m_BackpropOutputMatrixHeight * maxPoolToSave.m_BackpropOutputMatrixWidth; ++i)
	{
		output << maxPoolToSave.m_BackpropagationOutput[i] << " ";
	}

	output << endl;
	output.close();
}

void CunetaLogger::LogConvolutionState(Convolution convolutionToSave, string outputDirectory, string imageName, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "Convolution_";
	logFilePath += imageName;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);

	output << "CONVOLUTION" << endl;
	output << imageName << endl;
	output << iteration << endl;

	output << endl;
	output << "m_InputDims" << endl;
	output << convolutionToSave.m_InputMatrixHeight << endl;
	output << convolutionToSave.m_InputMatrixWidth << endl;

	output << endl;
	output << "m_FilterDim" << endl;
	output << convolutionToSave.m_FilterSize << endl;

	output << endl;
	output << "m_PaddingDim" << endl;
	output << convolutionToSave.m_PaddingSize << endl;

	output << endl;
	output << "m_OutputDims" << endl;
	output << convolutionToSave.m_OutputMatrixHeight << endl;
	output << convolutionToSave.m_OutputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropInputDims" << endl;
	output << convolutionToSave.m_BackpropInputMatrixHeight << endl;
	output << convolutionToSave.m_BackpropInputMatrixWidth << endl;

	output << endl;
	output << "m_PaddedInputDims" << endl;
	output << convolutionToSave.m_PaddedInputHeight << endl;
	output << convolutionToSave.m_PaddedInputWidth << endl;

	output << endl;
	output << "m_BackpropOutputDims" << endl;
	output << convolutionToSave.m_BackpropOutputMatrixHeight << endl;
	output << convolutionToSave.m_BackpropOutputMatrixWidth << endl;

	output << endl;
	output << "m_InputMatrix" << endl;
	for (int i = 0; i < convolutionToSave.m_InputMatrixHeight * convolutionToSave.m_InputMatrixWidth; ++i)
	{
		output << convolutionToSave.m_InputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_Filter" << endl;

	for (int i = 0; i < convolutionToSave.m_FilterSize * convolutionToSave.m_FilterSize; ++i)
	{
		output << convolutionToSave.m_Filter[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_FlippedFilter" << endl;

	for (int i = 0; i < convolutionToSave.m_FilterSize * convolutionToSave.m_FilterSize; ++i)
	{
		output << convolutionToSave.m_FlippedFilter[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_OutputMatrix" << endl;

	for (int i = 0; i < convolutionToSave.m_OutputMatrixHeight * convolutionToSave.m_OutputMatrixWidth; ++i)
	{
		output << convolutionToSave.m_OutputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_PaddedBackpropInput" << endl;

	for (int i = 0; i < convolutionToSave.m_PaddedInputHeight * convolutionToSave.m_PaddedInputWidth; ++i)
	{
		output << convolutionToSave.m_PaddedBackpropInput[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackPropInputMatrix" << endl;

	for (int i = 0; i < convolutionToSave.m_BackpropInputMatrixHeight * convolutionToSave.m_BackpropInputMatrixWidth; ++i)
	{
		output << convolutionToSave.m_BackPropInputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackpropagationOutput" << endl;

	for (int i = 0; i < convolutionToSave.m_BackpropOutputMatrixHeight * convolutionToSave.m_BackpropOutputMatrixWidth; ++i)
	{
		output << convolutionToSave.m_BackpropagationOutput[i] << " ";
	}

	output << endl;
	output.close();
}

void CunetaLogger::LogTransposeConvolutionState(TransposeConvolution transposeConvolutionToSave, string outputDirectory, string imageName, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "TransposeConvolution_";
	logFilePath += imageName;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);

	output << "TRANSPOSE_CONVOLUTION" << endl;
	output << imageName << endl;
	output << iteration << endl;

	output << endl;
	output << "m_InputDims" << endl;
	output << transposeConvolutionToSave.m_InputMatrixHeight << endl;
	output << transposeConvolutionToSave.m_InputMatrixWidth << endl;

	output << endl;
	output << "m_FilterDim" << endl;
	output << transposeConvolutionToSave.m_FilterSize << endl;

	output << endl;
	output << "m_PaddingDim" << endl;
	output << transposeConvolutionToSave.m_PaddingSize << endl;

	output << endl;
	output << "m_OutputDims" << endl;
	output << transposeConvolutionToSave.m_OutputMatrixHeight << endl;
	output << transposeConvolutionToSave.m_OutputMatrixWidth << endl;

	output << endl;
	output << "m_BackpropInputDims" << endl;
	output << transposeConvolutionToSave.m_BackpropInputMatrixHeight << endl;
	output << transposeConvolutionToSave.m_BackpropInputMatrixWidth << endl;


	output << endl;
	output << "m_BackpropOutputDims" << endl;
	output << transposeConvolutionToSave.m_BackpropOutputMatrixHeight << endl;
	output << transposeConvolutionToSave.m_BackpropOutputMatrixWidth << endl;

	output << endl;
	output << "m_InputMatrix" << endl;
	for (int i = 0; i < transposeConvolutionToSave.m_InputMatrixHeight * transposeConvolutionToSave.m_InputMatrixWidth; ++i)
	{
		output << transposeConvolutionToSave.m_InputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_Filter" << endl;

	for (int i = 0; i < transposeConvolutionToSave.m_FilterSize * transposeConvolutionToSave.m_FilterSize; ++i)
	{
		output << transposeConvolutionToSave.m_Filter[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_FlippedFilter" << endl;

	for (int i = 0; i < transposeConvolutionToSave.m_FilterSize * transposeConvolutionToSave.m_FilterSize; ++i)
	{
		output << transposeConvolutionToSave.m_FlippedFilter[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_OutputMatrix" << endl;

	for (int i = 0; i < transposeConvolutionToSave.m_OutputMatrixHeight * transposeConvolutionToSave.m_OutputMatrixWidth; ++i)
	{
		output << transposeConvolutionToSave.m_OutputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackPropInputMatrix" << endl;

	for (int i = 0; i < transposeConvolutionToSave.m_BackpropInputMatrixHeight * transposeConvolutionToSave.m_BackpropInputMatrixWidth; ++i)
	{
		output << transposeConvolutionToSave.m_BackPropInputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "m_BackpropagationOutput" << endl;

	for (int i = 0; i < transposeConvolutionToSave.m_BackpropOutputMatrixHeight * transposeConvolutionToSave.m_BackpropOutputMatrixWidth; ++i)
	{
		output << transposeConvolutionToSave.m_BackpropagationOutput[i] << " ";
	}

	output << endl;
	output.close();
}

void CunetaLogger::LogErrorState(ErrorCalcModule errorModuleToSave, string outputDirectory, string imageName, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "ErrorModule";
	logFilePath += imageName;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);

	output << "TRANSPOSE_CONVOLUTION" << endl;
	output << imageName << endl;
	output << iteration << endl;

	output << endl;
	output << "m_InputDims" << endl;
	output << errorModuleToSave.m_InputMatrixHeight << endl;
	output << errorModuleToSave.m_InputMatrixWidth << endl;

	output << endl;
	output << "m_OutputDims" << endl;
	output << errorModuleToSave.m_OutputMatrixHeight << endl;
	output << errorModuleToSave.m_OutputMatrixWidth << endl;

	output << endl;
	output << endl;
	output << "Network error" << endl;
	output << errorModuleToSave.networkError << endl;

	output << endl;
	output << "m_InputMatrix" << endl;
	for (int i = 0; i < errorModuleToSave.m_InputMatrixHeight * errorModuleToSave.m_InputMatrixWidth; ++i)
	{
		output << errorModuleToSave.m_InputMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "Sigmoid Output" << endl;

	for (int i = 0; i < errorModuleToSave.m_InputMatrixHeight * errorModuleToSave.m_InputMatrixHeight; ++i)
	{
		output << errorModuleToSave.sigmoidResultMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "Cross Entropy Output" << endl;

	for (int i = 0; i < errorModuleToSave.m_InputMatrixHeight * errorModuleToSave.m_InputMatrixHeight; ++i)
	{
		output << errorModuleToSave.crossEntropyResultMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "dLdx Output" << endl;

	for (int i = 0; i < errorModuleToSave.m_InputMatrixHeight * errorModuleToSave.m_InputMatrixHeight; ++i)
	{
		output << errorModuleToSave.dLdXMatrix[i] << " ";
	}

	output << endl;
	output << endl;
	output << "Intermeidate sums Output" << endl;

	for (int i = 0; i < errorModuleToSave.m_InputMatrixHeight; ++i)
	{
		output << errorModuleToSave.intermediateSumResult[i] << " ";
	}

	output << endl;
	output.close();
}


void CunetaLogger::AddImageNameToProcessingHistory(string outputDirectory, string imagePath, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "ProcessingHistory";
	logFilePath += ".cunetalog";

	std::ofstream outfile;

	bool logFileExists;
	ifstream file(logFilePath);
	if (file)
		logFileExists = true;
	else
		logFileExists = false;
	file.close();

	if (logFileExists)
	{
		outfile.open(logFilePath, std::ios_base::app); // append instead of overwrite
	}
	else {
		outfile.open(logFilePath);
	}

	outfile << iteration << imagePath << endl;
	outfile.close();
}

void CunetaLogger::AddErrorScore(string outputDirectory, float scoreToAdd, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "Error History";
	logFilePath += ".cunetalog";

	std::ofstream outfile;

	bool logFileExists;
	ifstream file(logFilePath);
	if (file)
		logFileExists = true;
	else
		logFileExists = false;
	file.close();

	if (logFileExists)
	{
		outfile.open(logFilePath, std::ios_base::app); // append instead of overwrite
	}
	else {
		outfile.open(logFilePath);
	}

	outfile << iteration << scoreToAdd << endl;
	outfile.close();
}

void CunetaLogger::SaveFilter(float* filter, int filterSize, string outputDirectory, string layer, int iteration)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "FilterSave_";
	logFilePath += layer;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cunetalog";

	ofstream output(logFilePath);

	output << filterSize << " " << filterSize << " ";

	for (int i = 0; i < filterSize * filterSize; ++i)
	{
		output << filter[i] << " ";
	}

	output << endl;
	output.close();
}


void CunetaLogger::SaveOutput(float* cunetaOutput, int height, int width ,string outputDirectory, string layer, int iteration, int ephoc)
{

	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "CUNETARES_";
	logFilePath += layer;
	logFilePath += "_";
	logFilePath += to_string(iteration);
	logFilePath += ".cuneta";

	ofstream output(logFilePath);

	output << height << " " << width << " ";

	for (int i = 0; i < height * width; ++i)
	{
		output << cunetaOutput[i] << " ";
	}

	output << endl;
	output.close();
}

void CunetaLogger::Save_RELU_Test(ReLU testSubject, string outputDirectory, int testNumber)
{
	string logFilePath;
	logFilePath += outputDirectory;
	logFilePath += "\\";
	logFilePath += "RELU_TEST_NUM_";
	logFilePath += to_string(testNumber);
	logFilePath += "_";
	logFilePath += to_string(testSubject.L_FORWARD_NumberOf_INPUTS);
	logFilePath += "_";
	logFilePath += to_string(testSubject.L_FORWARD_NumberOf_OUTPUTS);
	logFilePath += "_";
	logFilePath += to_string(testSubject.L_FORWARD_InputLayer_HEIGHT);
	logFilePath += "x";
	logFilePath += to_string(testSubject.L_FORWARD_InputLayer_WIDTH);
	logFilePath += ".cunetatest";

	ofstream output(logFilePath);
	output << testSubject.L_FORWARD_InputLayer_HEIGHT<< " " << testSubject.L_FORWARD_InputLayer_WIDTH << endl;
	output << "&&" << endl;
	output << testSubject.L_BACKWARD_InputLayer_HEIGHT << " " << testSubject.L_BACKWARD_InputLayer_WIDTH << endl;
	output << "&&" << endl;
	output << testSubject.L_FORWARD_OutputLayer_HEIGHT << " " << testSubject.L_FORWARD_OutputLayer_WIDTH << endl;
	output << "&&" << endl;
	output << testSubject.L_BACKWARD_OutputLayer_HEIGHT << " " << testSubject.L_BACKWARD_OutputLayer_WIDTH << endl;
	output << "&&" << endl;
	output << testSubject.L_FORWARD_NumberOf_INPUTS<< " " << testSubject.L_FORWARD_NumberOf_OUTPUTS <<endl;
	output << "&&"<<endl;
	output << testSubject.L_BACKWARD_NumberOf_INPUTS << " " << testSubject.L_BACKWARD_NumberOf_OUTPUTS <<endl;
	output << "$" << endl;

	//Forward pass inputs
	for (int i = 0; i < testSubject.L_FORWARD_NumberOf_INPUTS; ++i)
	{
		for (int j = 0; j < testSubject.L_FORWARD_InputLayer_HEIGHT * testSubject.L_FORWARD_InputLayer_WIDTH; j++)
		{
			output << testSubject.L_FORWARD_Pass_INPUTS[i][j] << " ";
		}
		output << endl;
		output << "*" <<endl;
	}

	output << "##" << endl; //Forward pass outputs

	for (int i = 0; i < testSubject.L_FORWARD_NumberOf_OUTPUTS; ++i)
	{
		for (int j = 0; j < testSubject.L_FORWARD_OutputLayer_HEIGHT * testSubject.L_FORWARD_OutputLayer_WIDTH; j++)
		{
			output << testSubject.L_FORWARD_Pass_OUTPUTS[i][j] << " ";
		}
		output << endl;
		output << "*" << endl;
	}

	output << "##" << endl; //Backward pass inputs

	for (int i = 0; i < testSubject.L_BACKWARD_NumberOf_INPUTS; ++i)
	{
		for (int j = 0; j < testSubject.L_BACKWARD_InputLayer_HEIGHT * testSubject.L_BACKWARD_InputLayer_WIDTH; j++)
		{
			output << testSubject.L_BACKWARD_Pass_INPUTS[i][j] << " ";
		}
		output << endl;
		output << "*" << endl;
	}

	output << "##" << endl; //Backward pass outputs

	for (int i = 0; i < testSubject.L_BACKWARD_NumberOf_OUTPUTS; ++i)
	{
		for (int j = 0; j < testSubject.L_BACKWARD_OutputLayer_HEIGHT * testSubject.L_BACKWARD_OutputLayer_WIDTH; j++)
		{
			output << testSubject.L_BACKWARD_Pass_OUTPUTS[0][j] << " ";
		}
		output << endl;
		output << "*" << endl;
	}


	output << endl;
	output.close();
}
