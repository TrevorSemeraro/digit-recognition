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

    Network(vector<int> layerSizes, ActivationFunction *hiddenLayerActivationFunction, ActivationFunction *outputLayerActivationFunction);

    int train(vector<vector<uint8_t>> batch_images, vector<uint8_t> batch_labels, double learnRate, double regularization, double momentum);

    vector<double> CalculateOutputs(vector<double> inputs);
    bool updateGradients(vector<double> image, double label, NetworkLearningData *learningData);

    double classify(vector<double> inputs);
    double test(vector<vector<uint8_t>> batch_images, vector<uint8_t> batch_labels);
};

#endif