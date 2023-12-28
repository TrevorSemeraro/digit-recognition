#ifndef LIB_H
#define LIB_H

#include <cstdint>
#include <vector>

using namespace std;

template <typename T>
vector<T> slice(vector<T> &arr, int X, int Y);

double getCost(vector<double> predictedValues, vector<double> expectedValues);
double getCostDerivative(double predictedValue, double expectedValue);

double RandomInNormalDistribution(double mean, double standardDeviation);

struct Parameters
{
	double initialLearningRate;
	double learnRateDecay;
	int minibatchSize;
	double momentum;
	double regularization;
};

class ActivationFunction
{
public:
	ActivationFunction() {}
	virtual double activation(vector<double> inputs, int i) = 0;
	virtual double derivative(vector<double> inputs, int i) = 0;
};

class SoftmaxActivation : public ActivationFunction
{
public:
	SoftmaxActivation(){};
	~SoftmaxActivation(){};
	double activation(vector<double> inputs, int i);
	double derivative(vector<double> inputs, int i);
};
class SigmoidActivation : public ActivationFunction
{
public:
	double activation(vector<double> inputs, int i);
	double derivative(vector<double> inputs, int i);
};
class ReLUActivation : public ActivationFunction
{
	double activation(vector<double> inputs, int i);
	double derivative(vector<double> inputs, int i);
};
class BinaryActivation : public ActivationFunction
{
	double activation(vector<double> inputs, int i);
	double derivative(vector<double> inputs, int i);
};
vector<double> generateExpectedValues(int label, int outputSize);
vector<double> formatImage(vector<uint8_t> image);
vector<double> convertArrToVector(double *arr, int size);

#endif