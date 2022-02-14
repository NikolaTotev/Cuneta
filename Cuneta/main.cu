
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "Convolution.cuh"
#include "Squishy.cuh"
#include "FolderManager.cuh"
#include "ImageIngester.cuh"
#include "Logger.cuh"
#include "SumBlock.cuh"
#include "Test_Utils.cuh"

using namespace std;

void Train(string _dataSetDirectory, int _numberOfEpochs, bool _verboseOutputEnabled);
void Test(string _testSaveDirectory);

int main()
{

	string dataSetDirectory = "";
	int numberOfEpocs = 0;
	bool enableVerboseOutput;
	bool trainingInputCompleted;
	string mode;
	
	cout << "Cuneta is starting..." << endl;
	cout << "Welcome, what would you like to start with?" << endl;

	string userInput;
	cin >> userInput;
	cout << endl;

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
		string saveDir;
		mode = "Test";

		cout << "Good idea, it's important to test things!" << endl;
		cout << endl;

		/*cout << "Where would you like to save the test results?" << endl;
		cin.ignore();
		getline(cin, saveDir);

		cout << endl;*/
		//cout << "Fantastic! Preparing test mode for you!" << endl;
		cout << "Preparing test mode for you!" << endl;
		cout << "Enjoy and good luck! May all of the tests pass! <3" << endl;
		cout << endl;
		NetworkValidator mrValidator = NetworkValidator(saveDir);
		mrValidator.TestFlowController();

		cout << endl;

		cout << "Oooo seems like the tests are done! How did they go?" << endl;
		cout << "Just as a reminder,  the test results can be found here:" << endl;
		cout << saveDir;

		cout << endl;

		cout << "That's all for now! See you later :3" << endl;
		
	}

	if(mode == "Train")
	{
		cout << "I'm done training!";
		cout << "You can view the results here.";
	}

	return 0;
}

void Train(string _dataSetDirectory, int _numberOfEpochs, bool _verboseOutputEnabled)
{

}