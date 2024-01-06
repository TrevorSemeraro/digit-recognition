#ifndef NETWORK_H
#define NETWORK_H

using namespace std;

#include <vector>
#include <cstdint>

#include "layer.h"
#include "learn.h"
#include "lib.h"

class Network
{
public:
    vector<Layer> layers;
    NetworkLearningData *learningData;

    Network(vector<int> layerSizes, ActivationFunction *hiddenLayerActivationFunction, ActivationFunction *outputLayerActivationFunction);

    int train(vector<LearnData> data_batch, double learnRate);

    vector<double> CalculateOutputs(vector<double> inputs);
    bool updateGradients(LearnData data, NetworkLearningData *learningData);

    double classify(vector<double> inputs);
    double test(vector<LearnData> test_data_batch);
};

#endif