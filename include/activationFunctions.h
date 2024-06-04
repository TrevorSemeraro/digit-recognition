#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>

using namespace std;

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

#endif