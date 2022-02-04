#include "ReLU.cuh"

void TestReLU(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight);
void TestMaxPool(float* inputMatrix, float* outputMatrix, int matrixWidth, int matrixHeight);
void TestConvolution(float* inputMatrix, int matrixHeight, int matrixWidth, int filterSize);