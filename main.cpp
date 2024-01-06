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

    vector<LearnData> training_data = createLearnData(dataset.training_images, dataset.training_labels);
    vector<LearnData> testing_data = createLearnData(dataset.test_images, dataset.test_labels);

    Parameters parameters = {
        0.05,
        64,
    };

    ActivationFunction *hiddenActivation = new SigmoidActivation();
    ActivationFunction *softmax = new SoftmaxActivation();

    int image_size = 28;
    int input_nodes = image_size * image_size;
    int output_nodes = 10;

    vector<int> layerSizes = {input_nodes, 100, output_nodes};

    cout << "Building Nueral Network of size (";
    for (int i = 0; i < (int)layerSizes.size(); i++)
    {
        cout << layerSizes[i] << " ";
    }
    cout << ")\n";

    Network *network = new Network(layerSizes, hiddenActivation, softmax);

    int epochs = 25;
    // int training_batches = (dataset.training_images.size() / parameters.minibatchSize);
    int training_batches = 100;

    int test_size = dataset.test_labels.size() / epochs;
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

        shuffle(begin(training_data), end(training_data), eng1);

        cout << "Shuffling training data\n";

        double sum_correct = 0;
        for (int mini_batch_index = 0; mini_batch_index < training_batches; mini_batch_index++)
        {

            int start_index = mini_batch_index * parameters.minibatchSize;
            int end_index = (mini_batch_index + 1) * parameters.minibatchSize;

            vector<LearnData> data_batch = slice(training_data, start_index, end_index);

            int num_correct = network->train(
                data_batch,
                currentLearningRate);

            sum_correct += num_correct;

            if (mini_batch_index % (training_batches / 4) == 0)
            {
                cout << "Batch " << mini_batch_index << "\n";
                cout << "Label: " << data_batch[0].label << "\n";
                cout << "Training: " << (double)num_correct / (parameters.minibatchSize) << "\n";
            }
        }
        cout << "Epoch Training " << epoch_index << ": " << sum_correct / (training_batches * parameters.minibatchSize) << "\n";

        currentLearningRate = (1.0 / (1 + 0.5 * epoch_index));

        int test_start_index = test_size * test_index;
        int test_end_index = test_size * (test_index + 1);
        // test_index++;

        vector<LearnData> test_data_batch = slice(testing_data, test_start_index, test_end_index);

        double testAccuracy = network->test(test_data_batch);

        cout << "Test Accuracy " << testAccuracy << "\n";

        saveNetwork(network, "network.txt");
    }

    return 0;
}