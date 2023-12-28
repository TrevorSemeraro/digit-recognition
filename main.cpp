#include <iostream>
#include <bitset>
#include <algorithm>
#include <vector>
#include <random>
#include <ctime>

#include "network.h"
#include "lib.h"
#include "save.h"

#include "mnist/mnist_reader.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    string MNIST_DATA_LOCATION = "./data/";
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "# of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "# of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "# of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "# of test labels = " << dataset.test_labels.size() << std::endl;

    Parameters parameters = {
        0.05,
        0.09,
        32,
        0.9,
        0.1};

    ActivationFunction *sigmoid = new ReLUActivation();
    ActivationFunction *softmax = new SoftmaxActivation();

    int image_size = 28;
    int input_nodes = image_size * image_size;
    int output_nodes = 10;

    vector<int> layerSizes = {input_nodes, 100, output_nodes};

    cout << "Building Nueral Network of size (";
    for(int i = 0; i < layerSizes.size(); i++) {
        cout << layerSizes[i] << " ";
    }
    cout << ")\n";

    Network *network = new Network(layerSizes, sigmoid, softmax);

    int epochs = 25;
    // int training_batches = (dataset.training_images.size() / parameters.minibatchSize) / 2;
    int training_batches = 101;
    
    int test_size = 250;
    int test_index = 0;

    double currentLearningRate = parameters.initialLearningRate;

    cout << "Training Nueral Network on " << training_batches << " batches.\n";

    for (int epoch_index = 0; epoch_index < epochs; epoch_index++)
    {
        // Random data shuffle (https://stackoverflow.com/questions/6926433/how-to-shuffle-a-stdvector)
        // Prevents training to fit data, makes network "learn"
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

        std::mt19937 eng1(seed);
        auto eng2 = eng1;

        shuffle(begin(dataset.training_images), end(dataset.training_images), eng1);
        shuffle(begin(dataset.training_labels), end(dataset.training_labels), eng2);

        double sum_correct = 0;
        for (int mini_batch_index = 0; mini_batch_index < training_batches; mini_batch_index++)
        {
            int start_index = mini_batch_index * parameters.minibatchSize;
            int end_index = (mini_batch_index + 1) * parameters.minibatchSize;

            vector<vector<uint8_t>> batch_images = slice(dataset.training_images, start_index, end_index);
            vector<uint8_t> batch_labels = slice(dataset.training_labels, start_index, end_index);

            int num_correct = network->train(
                batch_images,
                batch_labels,
                currentLearningRate,
                parameters.regularization,
                parameters.momentum);
            sum_correct += num_correct;

            if(mini_batch_index % 25 == 0) {
                // Makes sure the network is working / tells us how if its stuck
                cout << "Batch " << mini_batch_index << "\n";
            }
        }
        cout << "Epoch Training " << epoch_index << ": " << sum_correct / (training_batches * parameters.minibatchSize) << "\n";

        currentLearningRate = (1.0 / (1.0 + (parameters.learnRateDecay * epoch_index))) * parameters.initialLearningRate;

        int test_start_index = test_size * test_index;
        int test_end_index = test_size * (test_index + 1);
        // test_index++;

        vector<vector<uint8_t>> test_images = slice(dataset.test_images, test_start_index, test_end_index);
        vector<uint8_t> test_labels = slice(dataset.test_labels, test_start_index, test_end_index);

        double testAccuracy = network->test(test_images, test_labels);

        cout << "Testing " << epoch_index << ", " << testAccuracy << "\n";
    }

    saveNetwork(network, "network.txt");

    return 0;
}