#include <string>

#include "ReLU.cuh"
using namespace std;

void TestReLU(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies);
void TestBackpropReLU(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies);

void TestMaxPool(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, bool printMatricies);
void TestBackpropMaxPool(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, bool printMatricies);

void TestConvolution(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, int filterSize, int paddingSize, bool printMatricies);
void TestBackpropConvolution(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, int filterSize, int paddingSize, bool printMatricies);

void TestTransposeConvolution(int inputMatrixWidth, int inputMatrixHeight, int minInputVal, int maxInputVal, int filterSize, int paddingSize, bool printMatricies);
void TestBackpropTransposeConvolution(int inputMatrixWidth, int inputMatrixHeight, int fwdMinInputVal, int fwdMaxInputVal, int backMinInputVal, int backMaxInputVal, int filterSize, int paddingSize, bool printMatricies);

void TestBackpropErrorCalcModule(int inputMatrixWidth, int inputMatrixHeight, int rawMinInput, int rawMaxInput, int groundTruthMinInput, int groundTruthMaxInput, bool printMatricies);

void TestImageIngester(string inputPath, string groundTruthPath, bool printMatricies);