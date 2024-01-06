#ifndef LEARN_H
#define LEARN_H

using namespace std;

#include <vector>

class Layer;

struct LearnData {
    vector<double> image;
    vector<double> expectedOutputs;
    double label;
};

vector<LearnData> createLearnData(vector<vector<uint8_t>> images, vector<uint8_t> labels);

class LayerLearningData {
    public:
        double nodes_in;
        double nodes_out;
        vector<double> inputs;
        vector<double> weightedInputs;
        vector<double> activations;
        vector<double> nodeValues;

        LayerLearningData(Layer layer);
};

class NetworkLearningData {
    public:
        vector<LayerLearningData> layerData;

        NetworkLearningData();
};

#endif