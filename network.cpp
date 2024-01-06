#include <vector>
#include <iostream>
#include <bitset>
#include <ctime>
#include <math.h>

using namespace std;

#include "mnist/mnist_reader.hpp"

#include "network.h"
#include "lib.h"
#include "layer.h"
#include "learn.h"

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
int Network::train(vector<LearnData> data_batch, double learnRate)
{
    int num_correct = 0;
    for (int i = 0; i < (int)data_batch.size(); i++)
    {
        bool output = updateGradients(data_batch[i], learningData);
        if (output)
        {
            num_correct++;
        }
    }

    for (int i = 0; i < (int)layers.size(); i++)
    {
        Layer *currentLayer = &layers[i];
        currentLayer->ApplyGradients(learnRate / data_batch.size());
    }

    return num_correct;
}
double Network::classify(vector<double> inputs)
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

    return maxOutputNodeIndex;
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

bool Network::updateGradients(LearnData data, NetworkLearningData *learningData)
{
    // Feed data through nueral network to recieve output
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

    vector<double> output = input;
    double maxOutputNodeValue = 0;
    double maxOutputNodeIndex = 0;
    for (int i = 0; i < (int)output.size(); i++)
    {
        if (output[i] > maxOutputNodeValue)
        {
            maxOutputNodeValue = output[i];
            maxOutputNodeIndex = i;
        }
    }
    return (maxOutputNodeIndex == data.label);
}

double Network::test(vector<LearnData> test_data_batch)
{
    double sample_size = test_data_batch.size();
    double correct = 0;

    for (int i = 0; i < sample_size; i++)
    {
        double output = classify(test_data_batch[i].image);

        if (output == test_data_batch[i].label)
        {
            correct++;
        }
        else
        {
            // printImage(formattedImage);
            // cout << "Expected: " << (int)batch_labels[i] << " Recieved: " << output << "\n";
        }
    }

    double accuracy = (double)correct / (double)sample_size;
    return accuracy;
}