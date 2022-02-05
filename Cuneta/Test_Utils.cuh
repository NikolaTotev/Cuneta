#include <string>

#include "ReLU.cuh"
using namespace std;

void TestReLU(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight);
void TestMaxPool(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight, bool printMatricies);
void TestConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies);
void TestTransposeConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize, bool printMatricies);
void TestErrorCalcModule(float* inputMatrix, float* groundTruthMatrix, int matrixHeight, int matrixWidth, bool printMatricies);
void TestImageIngester(string inputPath, string groundTruthPath, bool printMatricies);