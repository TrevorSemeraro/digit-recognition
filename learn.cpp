using namespace std;

#include <vector>
#include <stdlib.h>
#include <cassert>

#include "learn.h"
#include "layer.h"
#include <iostream>

NetworkLearningData::NetworkLearningData(int len)
{
    layerData = new LayerLearningData *[len];
}

LayerLearningData::LayerLearningData(Layer layer)
{
    nodes_in = layer.nodes_in;
    nodes_out = layer.nodes_out;
    inputs = new double[layer.nodes_in];
    weightedInputs = new double[layer.nodes_out];
    activations = new double[layer.nodes_out];
    nodeValues = new double[layer.nodes_out];
}