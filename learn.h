#ifndef LEARN_H
#define LEARN_H

using namespace std;

#include <vector>

class Layer;

class LayerLearningData {
    public:
        double nodes_in;
        double nodes_out;
        double *inputs;
        double *weightedInputs;
        double *activations;
        double *nodeValues;

        LayerLearningData(Layer layer);
};

class NetworkLearningData {
    public:
        LayerLearningData **layerData;

        NetworkLearningData(int len);
};

#endif