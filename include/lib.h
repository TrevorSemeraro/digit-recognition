#ifndef LIB_H
#define LIB_H

#include <cstdint>
#include <vector>

#include "learn.h"

using namespace std;

vector<LearnData> mutateData(vector<LearnData> data);
vector<LearnData> slice(vector<LearnData> &arr, int X, int Y);

double getCost(vector<double> predictedValues, vector<double> expectedValues);
double getCostDerivative(double predictedValue, double expectedValue);

double RandomInNormalDistribution(double mean, double standardDeviation);

enum State
{
  MENU,
  TRAINING,
  TESTING
};

struct Parameters
{
	double initialLearningRate;
	int minibatchSize;
	double momentum;
	double learnRateDecay;
};

vector<double> generateExpectedValues(double label, int outputSize);
vector<double> formatImage(vector<uint8_t> image);
void printImage(vector<double> image);

#endif