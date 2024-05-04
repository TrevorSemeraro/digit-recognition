#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "learn.h"
#include "lib.h"

using namespace std;

class Layer
{
public:
    int nodes_in;
    int nodes_out;
    ActivationFunction *activationFunction;

    vector<vector<double>> weights;
    vector<vector<double>> cost_gradient_weights;
    vector<vector<double>> weight_velocity;
    vector<double> biases;
    vector<double> bias_velocity;
    vector<double> cost_gradient_biases;

    Layer(int _nodes_in, int _nodes_out, ActivationFunction *_activationFunction);
    void randomizeData();

    vector<double> calculateOutputs(vector<double> inputs);
    vector<double> calculateOutputs(vector<double> inputs, LayerLearningData*& learningData);

    void calculateOutputLayerNodeValues(LayerLearningData*& learningData, vector<double> expectedOutputs);
    void calculateHiddenLayerNodeValues(LayerLearningData*& learningData, Layer *prevLayer, LayerLearningData*& prevLayerNodeValues);
    void updateGradients(LayerLearningData*& learningData);
    void ApplyGradients(double learnRate, double momentumConstant);
};

#endif