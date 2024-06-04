#include <math.h>

#include <bitset>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

#include "../include/learn.h"
#include "../include/layer.h"
#include "../include/lib.h"

Layer::Layer(int _nodes_in, int _nodes_out, ActivationFunction* _activationFunction)
{
  nodes_in = _nodes_in;
  nodes_out = _nodes_out;
  activationFunction = _activationFunction;

  weights = {};
  weight_velocity = {};
  cost_gradient_weights = {};

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

    for (int j = 0; j < nodes_in; j++)
    {
      weightedInput += inputs[j] * weights[j][i];
      learningData->inputs[j] = inputs[j];
    }

    learningData->weightedInputs[i] = weightedInput;
    weighted_inputs.push_back(weightedInput);
  }

  for (int i = 0; i < (int)weighted_inputs.size(); i++)
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

  for (int i = 0; i < (int)weighted_inputs.size(); i++)
  {
    double outputValue = activationFunction->activation(weighted_inputs, i);
    activation_values.push_back(outputValue);
  }

  return activation_values;
}

void Layer::calculateOutputLayerNodeValues(LayerLearningData*& learningData, vector<double> expectedOutputs)
{
  for (int i = 0; i < learningData->nodes_out; i++)
  {
    double costDerivative = getCostDerivative(learningData->activations[i], expectedOutputs[i]);
    double activationDerivative = activationFunction->derivative(learningData->weightedInputs, i);

    learningData->nodeValues[i] = costDerivative * activationDerivative;
  }
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
    newNodeValue *= activationFunction->derivative(learningData->weightedInputs, newNodeIndex);
    learningData->nodeValues[newNodeIndex] = newNodeValue;
  }
}

void Layer::updateGradients(LayerLearningData*& learningData)
{
  for (int nodeOut = 0; nodeOut < nodes_out; nodeOut++)
  {
    double nodeValue = learningData->nodeValues[nodeOut];

    for (int nodeIn = 0; nodeIn < nodes_in; nodeIn++)
    {
      double deriveCostWRTWeight = learningData->inputs[nodeIn] * nodeValue;

      cost_gradient_weights[nodeIn][nodeOut] += deriveCostWRTWeight;
    }

    cost_gradient_biases[nodeOut] += nodeValue;
  }
}

void Layer::ApplyGradients(double learnRate, double momentumConstant)
{
  // double weightDecay = (1 - 0.01 * momentumConstant);
  double weightDecay = 1;

  for (int node_in_index = 0; node_in_index < nodes_in; node_in_index++)
  {
    for (int node_out_index = 0; node_out_index < nodes_out; node_out_index++)
    {
      // weights[node_in_index][node_out_index] += -(cost_gradient_weights[node_in_index][node_out_index] * learnRate);
      // https://optimization.cbe.cornell.edu/index.php?title=Momentum
      weight_velocity[node_in_index][node_out_index] = momentumConstant * weight_velocity[node_in_index][node_out_index] + cost_gradient_weights[node_in_index][node_out_index];

      weights[node_in_index][node_out_index] = weights[node_in_index][node_out_index] * weightDecay - learnRate * weight_velocity[node_in_index][node_out_index];

      cost_gradient_weights[node_in_index][node_out_index] = 0;
    }
  }

  for (int i = 0; i < nodes_out; i++)
  {
    // biases[i] += -cost_gradient_biases[i] * learnRate;
    bias_velocity[i] = momentumConstant * bias_velocity[i] + cost_gradient_biases[i];
    biases[i] = biases[i] - learnRate * bias_velocity[i];

    cost_gradient_biases[i] = 0;
  }
}
