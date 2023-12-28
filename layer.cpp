#include <vector>
#include <iostream>
#include <bitset>
#include <ctime>
#include <math.h>

using namespace std;

#include "layer.h"
#include "learn.h"
#include "lib.h"

Layer::Layer(int _nodes_in, int _nodes_out, ActivationFunction *_activationFunction)
{
    nodes_in = _nodes_in;
    nodes_out = _nodes_out;
    activationFunction = _activationFunction;

    weights = {};
    cost_gradient_weights = {};
    weight_velocity = {};
    biases = {};
    bias_velocity = {};
    cost_gradient_biases = {};

    randomizeData();
}

void Layer::randomizeData()
{
    for (int i = 0; i < nodes_in; i++)
    {
        vector<double> weights_row = {};
        vector<double> weights_cost_gradients = {};
        vector<double> weights_velocity_row = {};

        for (int j = 0; j < nodes_out; j++)
        {
            // double randomWeight = RandomInNormalDistribution(0.5, 0.5) / sqrt(nodes_in);
            double randomWeight = RandomInNormalDistribution(0, 1) / sqrt(nodes_in);

            if (isnan(randomWeight) || isinf(randomWeight))
            {
                cout << "Random Weight is NaN or Inf\n";
                randomWeight = 0;
            }

            weights_row.push_back(randomWeight);
            weights_cost_gradients.push_back(0);
            weights_velocity_row.push_back(0);
        }

        biases.push_back(0);
        weights.push_back(weights_row);
        cost_gradient_biases.push_back(0);
        bias_velocity.push_back(0);
        weight_velocity.push_back(weights_velocity_row);
        cost_gradient_weights.push_back(weights_cost_gradients);
    }
}

vector<double> Layer::calculateOutputs(vector<double> inputs, LayerLearningData*& learningData)
{
    vector<double> weighted_inputs = {};
    vector<double> activation_values = {};

    for (int i = 0; i < nodes_out; i++)
    {
        double weightedInput = biases[i];
        // cout << "Bias: (" << i << ")" << biases[i] << "\n";

        for (int j = 0; j < nodes_in; j++)
        {
            weightedInput += inputs[j] * weights[j][i];
            learningData->inputs[j] = inputs[j];
        }

        learningData->weightedInputs[i] = weightedInput;
        weighted_inputs.push_back(weightedInput);
    }

    for(int i = 0; i < weighted_inputs.size(); i++)
    {
        double outputValue = activationFunction->activation(weighted_inputs, i);

        learningData->activations[i] = outputValue;
        activation_values.push_back(outputValue);
    }

    return activation_values;
}

vector<double> Layer::calculateOutputs(vector<double> inputs)
{
    vector<double> weighted_inputs = {};

    for (int i = 0; i < nodes_out; i++)
    {
        double weightedInput = biases[i];
        // cout << "Bias: (" << i << ")" << biases[i] << "\n";

        for (int j = 0; j < nodes_in; j++)
        {
            weightedInput += inputs[j] * weights[j][i];
        }

        weighted_inputs.push_back(weightedInput);
    }

    vector<double> activation_values = {};

    for(int i = 0; i < weighted_inputs.size(); i++)
    {
        double outputValue = activationFunction->activation(weighted_inputs, i);
        activation_values.push_back(outputValue);
    }

    return activation_values;
}

void Layer::calculateOutputLayerNodeValues(LayerLearningData*& learningData, vector<double> expectedOutputs)
{
    int length = learningData->nodes_out;

    for (int i = 0; i < length; i++)
    {
        double costDerivative = getCostDerivative(learningData->activations[i], expectedOutputs[i]);

        vector<double> inputs = convertArrToVector(learningData->inputs, learningData->nodes_in);
        double activationDerivative = activationFunction->derivative(inputs, i);

        learningData->nodeValues[i] = costDerivative * activationDerivative;

    }

    return;
}
void Layer::calculateHiddenLayerNodeValues(LayerLearningData*& learningData, Layer* prevLayer, LayerLearningData*& prevLayerData)
{
    for (int newNodeIndex = 0; newNodeIndex < nodes_out; newNodeIndex++)
    {
        double newNodeValue = 0;

        for (int oldNodeIndex = 0; oldNodeIndex < prevLayerData->nodes_out; oldNodeIndex++)
        {
            double weightedInputDerivative = prevLayer->weights[newNodeIndex][oldNodeIndex];

            newNodeValue += weightedInputDerivative * prevLayerData->nodeValues[oldNodeIndex];
        }

        vector<double> inputs = convertArrToVector(learningData->inputs, learningData->nodes_in);
        
        newNodeValue *= activationFunction->derivative(inputs, newNodeIndex);
        learningData->nodeValues[newNodeIndex] = newNodeValue;
    }
}

void Layer::updateGradients(LayerLearningData*& learningData)
{
    for (int i = 0; i < nodes_out; i++)
    {
        double nodeValue = learningData->nodeValues[i];

        for (int j = 0; j < nodes_in; j++)
        {
            double deriveCostWRTWeight = nodeValue * learningData->inputs[j];

            cost_gradient_weights[j][i] += deriveCostWRTWeight;
        }

        cost_gradient_biases[i] += nodeValue;
    }
}

void Layer::ApplyGradients(double learnRate, double regularization, double momentum)
{
    double weightDecay = (1 - regularization * learnRate);

    for (int node_in_index = 0; node_in_index < nodes_in; node_in_index++)
    {
        for (int node_out_index = 0; node_out_index < nodes_out; node_out_index++)
        {
            double weight = weights[node_in_index][node_out_index];
            double velocity = weight_velocity[node_in_index][node_out_index] * momentum - cost_gradient_weights[node_in_index][node_out_index] * learnRate;
            weight_velocity[node_in_index][node_out_index] = velocity;

            weights[node_in_index][node_out_index] = weight * weightDecay + velocity;
            cost_gradient_weights[node_in_index][node_out_index] = 0;
        }
    }

    for (int i = 0; i < nodes_out; i++)
    {
        double velocity = bias_velocity[i] * momentum - cost_gradient_biases[i] * learnRate;
        bias_velocity[i] = velocity;
        biases[i] += velocity;
        cost_gradient_biases[i] = 0;
    }
}