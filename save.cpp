#include <fstream>
#include <iostream>

#include "network.h"
#include "save.h"

// #include "nlohmann/json.hpp"

// using json = nlohmann::json;

using namespace std;

void saveNetwork(Network *network, string filename)
{
    // ofstream file;
    // file.open(filename);

    // json data = {
    //     {"layers", {784, 128, 10}},
    //     {"weights", {
    //                     {},
    //                     {},
    //                     {},
    //                 }},
    //     {"biases", {
    //                    {},
    //                    {},
    //                    {},
    //                }}};

    // // file << "{" << network->layers[0].nodes_in;
    // // for (int i = 0; i < network->layers.size(); i++)
    // // {
    // //     file << "," << network->layers[i].nodes_out;
    // // }
    // // file << "}";

    // // for (int i = 0; i < network->layers.size(); i++)
    // // {
    // //     Layer layer = network->layers[i];

    // //     file << "{";

    // //     for (int j = 0; j < layer.nodes_in; j++)
    // //     {
    // //         for (int k = 0; k < layer.nodes_out; k++)
    // //         {
    // //             file << layer.weights[j][k] << ",";
    // //         }
    // //     }
    // //     file << "|";

    // //     for (int j = 0; j < layer.nodes_out; j++)
    // //     {
    // //         file << layer.biases[j] << ",";
    // //     }

    // //     file << "}";
    // // }

    // file << data.dump();
    // file.close();
}

Network loadNetwork(string filename)
{
    // Import the network from the file
    ActivationFunction *sigmoid = new SigmoidActivation();
    ActivationFunction *softmax = new SoftmaxActivation();

    // std::ifstream infile(filename);

    // json data = json::parse(infile);

    // data['layers']

    Network n = Network({784, 128, 10}, sigmoid, softmax);

    return n;
}