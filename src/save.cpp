
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

#include "../include/save.h"
#include "../include/network.h"

using json = nlohmann::json;

void saveNetwork(Network *network, string filename)
{
  ofstream file;
  file.open(filename);

  json layers_array = json::array();
  json weights_array = json::array();
  json bias_array = json::array();

  for (auto layer : network->layers)
  {
    layers_array.push_back(layer.nodes_in);

    for (int j = 0; j < layer.nodes_in; j++)
    {
      for (int k = 0; k < layer.nodes_out; k++)
      {
        weights_array.push_back(layer.weights[j][k]);
      }
    }

    for (int j = 0; j < layer.nodes_out; j++)
    {
      bias_array.push_back(layer.biases[j]);
    }
  }

  layers_array.push_back(network->layers[network->layers.size() - 1].nodes_out);

  json data = {
      {"layers", layers_array},
      {"weights", weights_array},
      {"biases", bias_array},
  };

  file << data.dump();
  file.close();
}

Network *loadNetwork(string filename)
{
  ActivationFunction *sigmoid = new SigmoidActivation();
  ActivationFunction *softmax = new SoftmaxActivation();

  std::ifstream infile(filename);

  json data = json::parse(infile);

  vector<int> layers = data["layers"];

  // Print the layers to the console
  cout << "Layers: ";
  for (int i = 0; i < layers.size(); i++)
  {
    cout << layers[i] << " ";
  }
  cout << endl;

  Network *n = new Network(layers, sigmoid, softmax);

  vector<double> weights = data["weights"];
  vector<double> biases = data["biases"];

  int weight_index = 0;
  int bias_index = 0;

  for (int i = 0; i < n->layers.size(); i++)
  {
    for (int j = 0; j < n->layers[i].nodes_in; j++)
    {
      for (int k = 0; k < n->layers[i].nodes_out; k++)
      {
        n->layers[i].weights[j][k] = weights[weight_index];
        weight_index++;
      }
    }

    for (int j = 0; j < n->layers[i].nodes_out; j++)
    {
      n->layers[i].biases[j] = biases[bias_index];
      bias_index++;
    }
  }

  return n;
}