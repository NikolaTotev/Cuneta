
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "FolderManager.cuh"
#include "ImageIngester.cuh"
#include "Logger.cuh"
#include "SumBlock.cuh"
#include "Test_Utils.cuh"
using namespace std;

void Train(string _dataSetDirectory, int _numberOfEpochs, bool _verboseOutputEnabled);

int main()
{

	string dataSetDirectory = "";
	int numberOfEpocs = 0;
	bool enableVerboseOutput;
	bool trainingInputCompleted;
	string mode;

	Convolution conv = Convolution(3, 2, 4, 2, 6, 4);

	float** inputs = new float* [4];

	for (int j = 0; j < 4; ++j)
	{
		inputs[j] = new float[4 * 6];
		for (int i = 0; i < 4 * 6; ++i)
		{
			inputs[j][i] = 1;
		}
	}

	float** back_inputs = new float* [2];

	for (int j = 0; j < 2; ++j)
	{
		back_inputs[j] = new float[4 * 2];
		for (int i = 0; i < 2 * 4; ++i)
		{
			back_inputs[j][i] = 4;
		}
	}


	conv.LayerForwardPass(inputs);
	conv.LayerBackwardPass(back_inputs);
	int counter = 1;

	cout << "Forward inputs" << endl;

	for (int j = 0; j < conv.L_FORWARD_NumberOf_INPUTS; ++j)
	{
		for (int i = 0; i < conv.L_FORWARD_InputLayer_HEIGHT * conv.L_FORWARD_InputLayer_WIDTH; ++i)
		{
			cout << conv.L_FORWARD_Pass_INPUTS[j][i];
			counter++;
			if(counter == conv.L_FORWARD_InputLayer_WIDTH+1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}

	cout << "Forward filters" << endl;

	for (int j = 0; j < conv.L_NumberOf_FILTERS; ++j)
	{
		for (int i = 0; i < conv.m_FilterSize * conv.m_FilterSize; ++i)
		{
			cout << conv.L_Filters[j][i] << " ";
			counter++;
			if (counter == conv.m_FilterSize + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}

	cout << endl;
	cout << "Forward Outputs" << endl;
	counter = 1;

	for (int j = 0; j < conv.L_FORWARD_NumberOf_OUTPUTS; j++)
	{
		for (int i = 0; i < conv.L_FORWARD_OutputLayer_HEIGHT * conv.L_FORWARD_OutputLayer_WIDTH; ++i)
		{
			cout << conv.L_FORWARD_Pass_OUTPUTS[j][i] << " ";
			counter++;
			if (counter == conv.L_FORWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}
	cout << endl;
	cout << "Backpass inputs" << endl;
	counter = 1;

	for (int j = 0; j < conv.L_FORWARD_NumberOf_OUTPUTS; ++j)
	{
		for (int i = 0; i < conv.L_BACKWARD_InputLayer_HEIGHT * conv.L_BACKWARD_InputLayer_WIDTH; ++i)
		{
			cout << conv.L_BACKWARD_Pass_INPUTS[j][i] << " ";
			counter++;
			if (counter == conv.L_BACKWARD_InputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}


	cout << endl;
	cout << "Backpass Outputs" << endl;
	counter = 1;

	for (int j = 0; j < conv.L_FORWARD_NumberOf_INPUTS; ++j)
	{
		for (int i = 0; i < conv.L_BACKWARD_OutputLayer_HEIGHT * conv.L_BACKWARD_OutputLayer_WIDTH; ++i)
		{
			cout << conv.L_BACKWARD_Pass_OUTPUTS[j][i] << " ";
			counter++;
			if (counter == conv.L_BACKWARD_OutputLayer_WIDTH + 1)
			{
				cout << endl;
				counter = 1;
			}
		}
		cout << endl;
	}


	/*cout << "Cuneta is starting..." << endl;
	cout << "Welcome, what would you like to start with?";

	string userInput;
	cin >> userInput;

	if (userInput == "Train")
	{
		mode = "Train";
		cout << "Great! I love learning new things!" << endl;
		cout << "Please enter a path to a Cuneta compatible data set." << endl;
		cin >> dataSetDirectory;
		cout << endl;
		cout << endl;

		cout << "Thank you! I'm almost ready to start learning, I just need know a few more things." << endl;
		cout << "How many epochs do you want me to train for?"<<endl;
		cin >> numberOfEpocs;
		cout << endl;
		cout << endl;

		while(numberOfEpocs <=0)
		{
			cout << "Hmm, I can't work with that number. Please enter a number larger than 1.";
			cin >> numberOfEpocs;
			cout << endl;
			cout << endl;
		}
		cout << "That's a greate number!";

		cout << "Would you like me to provide a verbose output of my progress? (Y/N)";
		cin >> userInput;
		cout << endl;
		cout << endl;

		enableVerboseOutput = userInput == "Y";

		cout << "Great! That's all of the information that I need." << endl;
		cout << "To recap:" << endl;
		cout << "I'll be using the data set located here: " << endl;
		cout << dataSetDirectory << endl;


		cout << "I'm learning for " << numberOfEpocs << "epochs"<<endl;
		if(enableVerboseOutput)
		{
			cout << "and you would like me to provide a verbose output." << endl;
		}
		else
		{
			cout << "and you don't want me to provide a verbose output." << endl;
		}

		cout << endl;
		cout << endl;

		cout << "Does this sound good? (Y/N)";
		cin >> userInput;
		cout << endl;
		cout << endl;

		if(userInput == "Y")
		{
			cout << "Excellent! Starting training."<<endl;
			Train(dataSetDirectory, numberOfEpocs, enableVerboseOutput);
		}
		else
		{
			trainingInputCompleted = false;

			while(!trainingInputCompleted)
			{
				cout << "Whops! I see something seems wrong. What parameter would you like to adjust?" << endl;
				cout << "The available options are: " << endl;
				cout << "Directory - to change the data set directory." << endl;
				cout << "Epoch - to change the number of epochs." << endl;
				cout << "Output - to change the verbose output flag." << endl;
				cout << "All - to change all of the parameters." << endl;
				cout << "You can also exit from the program by entering 'Cancel' or Ctrl+C";
				cin >> userInput;

				if (userInput == "Directory")
				{
					cout << "What directory would you like to switch to?" << endl;
					cin >> dataSetDirectory;
					cout << endl;
					cout << endl;
				}

				if (userInput == "Epoch")
				{
					cout << "How many epochs would you like me to learn? Right now I'll learn for: "<< numberOfEpocs << endl;
					cin >> numberOfEpocs;
					cout << endl;
					cout << endl;
				}

				if (userInput == "Output")
				{
					cout << "Would you like me to provide verbose output?" << endl;
					cin >> userInput;
					enableVerboseOutput = userInput == "Y";
					cout << endl;
					cout << endl;
				}

				if(userInput == "All")
				{
					cout << "What directory would you like to switch to?" << endl;
					cin >> dataSetDirectory;
					cout << endl;
					cout << endl;

					cout << "How many epochs would you like me to learn? Right now I'll learn for: " << numberOfEpocs << endl;
					cin >> numberOfEpocs;
					cout << endl;
					cout << endl;

					cout << "Would you like me to provide verbose output?" << endl;
					cin >> userInput;
					enableVerboseOutput = userInput == "Y";
					cout << endl;
					cout << endl;
				}

				cout << "Great! I've updated the settings, are you ready to start?" << endl;
				cin >> userInput;

				trainingInputCompleted = userInput == "Y";
			}

			cout << "Great! Starting training." << endl;
			Train(dataSetDirectory, numberOfEpocs, enableVerboseOutput);

		}
	}

	if (userInput == "Test")
	{

	}

	if(mode == "Train")
	{
		cout << "I'm done training!";
		cout << "You can view the results here.";
	}*/

	return 0;
}

void Train(string _dataSetDirectory, int _numberOfEpochs, bool _verboseOutputEnabled)
{

}