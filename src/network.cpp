#include <math.h>

#include <bitset>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

#include "../include/layer.h"
#include "../include/learn.h"
#include "../include/lib.h"
#include "mnist/mnist_reader.hpp"
#include "../include/network.h"

Network::Network(vector<int> layerSizes, ActivationFunction *hiddenLayerActivationFunction, ActivationFunction *outputLayerActivationFunction)
{
  // Providing a seed value
  srand((unsigned)time(NULL));
  NetworkLearningData *learningData = new NetworkLearningData();

  for (int i = 0; i < (int)layerSizes.size() - 1; i++)
  {
    if (i == layerSizes.size() - 2)
    {
      // Output Layer
      Layer *newLayer = new Layer(layerSizes[i], layerSizes[i + 1], outputLayerActivationFunction);
      layers.push_back(*newLayer);
    }
    else
    {
      Layer *newLayer = new Layer(layerSizes[i], layerSizes[i + 1], hiddenLayerActivationFunction);
      layers.push_back(*newLayer);
    }

    LayerLearningData *currentLayer = new LayerLearningData(layers[i]);
    learningData->layerData.push_back(*currentLayer);
  }
  this->learningData = learningData;
}
TrainResponse Network::train(vector<LearnData> data_batch, double learnRate, double momentum)
{
  int num_correct = 0;
  double averageCost = 0;
  for (int i = 0; i < (int)data_batch.size(); i++)
  {
    LearnData currentLearnDataBlob = data_batch[i];
    vector<double> outputs = updateGradients(currentLearnDataBlob);

    double maxOutputNodeValue = 0;
    double maxOutputNodeIndex = 0;
    for (int i = 0; i < (int)outputs.size(); i++)
    {
      if (outputs[i] > maxOutputNodeValue)
      {
        maxOutputNodeValue = outputs[i];
        maxOutputNodeIndex = i;
      }
    }
    if (maxOutputNodeIndex == currentLearnDataBlob.label)
    {
      num_correct++;
    }

    averageCost += getCost(outputs, generateExpectedValues(currentLearnDataBlob.label, 10));
  }

  for (int i = 0; i < (int)layers.size(); i++)
  {
    Layer *currentLayer = &layers[i];
    currentLayer->ApplyGradients(learnRate / data_batch.size(), momentum);
  }

  averageCost /= data_batch.size();

  return {
    num_correct,
    averageCost,
  };
}

vector<double> Network::updateGradients(LearnData data)
{
  vector<double> input = data.image;

  for (unsigned int i = 0; i < layers.size(); i++)
  {
    Layer *currentLayer = &layers[i];
    LayerLearningData *ld = &learningData->layerData[i];

    input = currentLayer->calculateOutputs(input, ld);
  }

  int outputLayerIndex = layers.size() - 1;
  Layer *outputLayer = &layers[outputLayerIndex];
  LayerLearningData *outputLayerData = &learningData->layerData[outputLayerIndex];

  outputLayer->calculateOutputLayerNodeValues(outputLayerData, data.expectedOutputs);
  outputLayer->updateGradients(outputLayerData);

  for (int i = outputLayerIndex - 1; i >= 0; i--)
  {
    Layer *currentLayer = &layers[i];
    LayerLearningData *currentLayerData = &learningData->layerData[i];

    Layer *nextLayer = &layers[i + 1];
    LayerLearningData *nextLayerData = &learningData->layerData[i + 1];

    currentLayer->calculateHiddenLayerNodeValues(currentLayerData, nextLayer, nextLayerData);
    currentLayer->updateGradients(currentLayerData);
  }

  return input;
}

ClassifyResponse Network::classify(vector<double> inputs)
{
  vector<double> outputs = CalculateOutputs(inputs);

  double maxOutputNodeValue = 0;
  double maxOutputNodeIndex = 0;

  for (int i = 0; i < (int)outputs.size(); i++)
  {
    if (outputs[i] > maxOutputNodeValue)
    {
      maxOutputNodeValue = outputs[i];
      maxOutputNodeIndex = i;
    }
  }

  return {
      outputs,
      maxOutputNodeIndex,
  };
}

vector<double> Network::CalculateOutputs(vector<double> inputs)
{
  vector<double> output = inputs;

  for (unsigned int i = 0; i < layers.size(); i++)
  {
    Layer currentLayer = layers[i];
    output = currentLayer.calculateOutputs(output);
  }

  return output;
}

double Network::test(vector<LearnData> test_data_batch)
{
  double sample_size = test_data_batch.size();
  double correct = 0;

  for (int i = 0; i < sample_size; i++)
  {
    ClassifyResponse response = classify(test_data_batch[i].image);
    double output = response.predicted;

    if (output == test_data_batch[i].label)
    {
      correct++;
    }
  }

  double accuracy = (double)correct / (double)sample_size;
  return accuracy;
}


#include <emscripten/emscripten.h>

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

EMSCRIPTEN_BINDINGS(my_class_example) {
    class_<Network>("Network")
        .constructor<vector<int> layerSizes, ActivationFunction *hiddenLayerActivationFunction, ActivationFunction *outputLayerActivationFunction>()
        .function("classify", &Network::classify)
        ;
