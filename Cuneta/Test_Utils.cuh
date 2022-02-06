#include <string>

#include "ReLU.cuh"
using namespace std;

void TestReLU(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies);
void TestBackpropReLU(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies);
void TestMaxPool(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight, bool printMatricies);
void TestBackpropMaxPool();
void TestConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies);
void TestBackpropConvolution();
void TestBackpropTransposeConvolution();
void TestErrorCalcModule();
void TestBackpropErrorCalcModule(float* inputMatrix, float* groundTruthMatrix, int matrixHeight, int matrixWidth, bool printMatricies);
void TestImageIngester(string inputPath, string groundTruthPath, bool printMatricies);