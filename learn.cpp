using namespace std;

#include <vector>
#include <stdlib.h>
#include <cassert>
#include <stdint.h>
#include <iostream>

#include "learn.h"
#include "layer.h"
#include "lib.h"

vector<LearnData> createLearnData(vector<vector<uint8_t>> images, vector<uint8_t> labels) {
    vector<LearnData> data;
    
    if(images.size() != labels.size()){
        cout << "Training batch sizes do not match for creating learning data.\n";
        return data;
    }

    for(int i = 0; i < (int) images.size(); i++){
        LearnData newData;
        newData.image = formatImage(images[i]);
        newData.label = labels[i];
        newData.expectedOutputs = generateExpectedValues(labels[i], 10);
        data.push_back(newData);
    }
    
    return data;
}

vector<double> populateVector(int size) {
    vector<double> vec;
    for(int i = 0; i < size; i++) {
        vec.push_back(0);
    }
    return vec;
}

NetworkLearningData::NetworkLearningData(){
    layerData = {};
    return;
}

LayerLearningData::LayerLearningData(Layer layer)
{
    nodes_in = layer.nodes_in;
    nodes_out = layer.nodes_out;

    inputs = populateVector(layer.nodes_in);
    weightedInputs = populateVector(layer.nodes_out);
    activations = populateVector(layer.nodes_out);
    nodeValues = populateVector(layer.nodes_out);
}