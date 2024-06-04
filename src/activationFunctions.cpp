
#include "../include/activationFunctions.h"
#include <math.h>

double SoftmaxActivation::activation(vector<double> inputs, int i)
{
    double sum = 0;

    for (int j = 0; j < (int)inputs.size(); j++)
    {
        sum += exp(inputs[j]);
    }

    double res = exp(inputs[i]) / sum;

    return res;
}

double SoftmaxActivation::derivative(vector<double> inputs, int i)
{
    double expSum = 0;
    for (int j = 0; j < (int)inputs.size(); j++)
    {
        expSum += exp(inputs[j]);
    }

    double ex = exp(inputs[i]);

    return (ex * expSum - ex * ex) / (expSum * expSum);
}

double SigmoidActivation::activation(vector<double> inputs, int i)
{
    return 1.0 / (1 + exp(-inputs[i]));
}

double SigmoidActivation::derivative(vector<double> inputs, int i)
{
    double a = activation(inputs, i);
    return a * (1 - a);
}

const double reluAlpha = 0.1;
double ReLUActivation::activation(vector<double> inputs, int i)
{
    if (inputs[i] > 0)
    {
        return inputs[i];
    }
    else
    {
        return reluAlpha * inputs[i];
    }
}

double ReLUActivation::derivative(vector<double> inputs, int i)
{
    if (inputs[i] > 0)
    {
        return 1;
    }
    else
    {
        return reluAlpha;
    }
}