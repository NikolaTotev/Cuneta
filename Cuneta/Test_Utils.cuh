#include <string>
#include "Convolution.cuh"
#include "ErrorCalcModule.cuh"
#include "ImageIngester.cuh"
#include "Logger.cuh"
#include "MaxPool.cuh"
#include "ReLU.cuh"
#include "TransposeConvolution.cuh"

using namespace std;
class NetworkValidator
{
	string testSaveDirectory = "";

public:
	NetworkValidator(string _saveDirectory);
	void TestFlowController();
	void TestReLU();
	void TestMaxPool();
	void TestConvolution();
	void TestTransposeConvolution();
	void SquishTest();
	void TestSumBlock();
	void TestErrorBlock();
};