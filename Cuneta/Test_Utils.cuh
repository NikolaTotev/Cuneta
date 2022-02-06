#include <string>

#include "ReLU.cuh"
using namespace std;

void TestReLU(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies);
void TestBackpropReLU(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies);
void TestMaxPool(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies);
void TestBackpropMaxPool(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies);
void TestConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies);
void TestBackpropConvolution();
void TestBackpropTransposeConvolution();
void TestErrorCalcModule();
void TestBackpropErrorCalcModule(float* inputMatrix, float* groundTruthMatrix, int matrixHeight, int matrixWidth, bool printMatricies);
void TestImageIngester(string inputPath, string groundTruthPath, bool printMatricies);