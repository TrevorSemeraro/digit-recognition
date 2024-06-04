#ifndef NETWORK_H
#define NETWORK_H

using namespace std;

#include <cstdint>
#include <vector>

#include "layer.h"
#include "learn.h"
#include "lib.h"

struct ClassifyResponse
{
  vector<double> outputs;
  double predicted;
};

struct TrainResponse {
  int num_correct;
  double cost;
};

class Network
{
 public:
  vector<Layer> layers;
  NetworkLearningData *learningData;

  Network(vector<int> layerSizes, ActivationFunction *hiddenLayerActivationFunction, ActivationFunction *outputLayerActivationFunction);
  TrainResponse train(vector<LearnData> data_batch, double learnRate, double momentum);
  vector<double> CalculateOutputs(vector<double> inputs);
  vector<double> updateGradients(LearnData data);
  ClassifyResponse classify(vector<double> inputs);
  double test(vector<LearnData> test_data_batch);
};

#endif