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

    for (unsigned int i = 0; i < layerSizes.size() - 1; i++)
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
    }
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

bool Network::updateGradients(vector<double> image, double label, NetworkLearningData *learningData)
{
    // Feed data through nueral network to recieve output
    vector<double> input = image;

    for (unsigned int i = 0; i < layers.size(); i++)
    {
        Layer *currentLayer = &layers[i];
        LayerLearningData *currentLayerData = learningData->layerData[i];

        input = currentLayer->calculateOutputs(input, currentLayerData);
    }

    vector<double> output = input;
    double maxOutputNodeValue = 0;
    double maxOutputNodeIndex = 0;
    for(int i = 0; i < output.size(); i++)
    {
        if(output[i] > maxOutputNodeValue)
        {
            maxOutputNodeValue = output[i];
            maxOutputNodeIndex = i;
        }
    }

    // Backpropagate through network to calculate cost gradients
    int outputLayerIndex = layers.size() - 1;

    Layer *outputLayer = &layers[outputLayerIndex];
    LayerLearningData *outputLayerData = learningData->layerData[outputLayerIndex];

    vector<double> expectedOutputs = generateExpectedValues((int)label, outputLayer->nodes_out);

    outputLayer->calculateOutputLayerNodeValues(outputLayerData, expectedOutputs);
    outputLayer->updateGradients(outputLayerData);

    for (int i = outputLayerIndex - 1; i >= 0; i--)
    {
        Layer *currentLayer = &layers[i];
        LayerLearningData *currentLayerData = learningData->layerData[i];

        Layer *nextLayer = &layers[i + 1];
        LayerLearningData *nextLayerData = learningData->layerData[i + 1];

        currentLayer->calculateHiddenLayerNodeValues(currentLayerData, nextLayer, nextLayerData);
        currentLayer->updateGradients(currentLayerData);
    }
    // cout << maxOutputNodeIndex << " " << maxOutputNodeValue << " " << label << "\n";
    return (maxOutputNodeIndex == label);
}

int Network::train(vector<vector<uint8_t>> batch_images, vector<uint8_t> batch_labels, double learnRate, double regularization, double momentum)
{
    if (batch_images.size() != batch_labels.size())
    {
        cout << "Training batch sizes do not match.\n";
        return 0;
    }

    NetworkLearningData *learningData = new NetworkLearningData(layers.size());
    for (int i = 0; i < (int)layers.size(); i++)
    {
        LayerLearningData *layerData = new LayerLearningData(layers[i]);
        learningData->layerData[i] = layerData;
    }

    int num_correct = 0;
    for (int i = 0; i < (int)batch_images.size(); i++)
    {
        vector<double> formattedInput = formatImage(batch_images[i]);
        bool output = updateGradients(formattedInput, (double)batch_labels[i], learningData);
        if(output) {
            num_correct++;
        }
    }

    // Apply Gradients
    for (int i = 0; i < (int)layers.size(); i++)
    {
        Layer *currentLayer = &layers[i];
        currentLayer->ApplyGradients(learnRate / batch_labels.size(), regularization, momentum);
    }

    return num_correct;
}

double Network::test(vector<vector<uint8_t>> batch_images, vector<uint8_t> batch_labels)
{
    if (batch_images.size() != batch_labels.size())
    {
        cout << "Training batch sizes do not match.\n";
        return 0;
    }

    double sample_size = batch_images.size();
    double correct = 0;

    for (int i = 0; i < sample_size; i++)
    {

        vector<double> formattedImage = {};
        vector<uint8_t> image = batch_images[i];

        for (unsigned int i = 0; i < image.size(); i++)
        {
            formattedImage.push_back((double)image[i] / 255);
        }
        double output = classify(formattedImage);

        if (output == (int)batch_labels[i])
        {
            correct++;
        }
    }

    double accuracy = (double)correct / (double)sample_size;
    return accuracy;
}