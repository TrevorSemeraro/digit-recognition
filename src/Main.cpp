#include <algorithm>
#include <bitset>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "../include/activationFunctions.h"
#include "../include/draw.h"
#include "../include/lib.h"
#include "../include/network.h"
#include "../include/save.h"
#include "../lib/mnist/mnist_reader.hpp"
#include "Platform/Platform.hpp"

int main()
{
  //
  // Load MNIST Data
  //
  string MNIST_DATA_LOCATION = "./data/";

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

  std::cout << "# of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "# of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "# of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "# of test labels = " << dataset.test_labels.size() << std::endl;

  vector<LearnData> training_data = createLearnData(dataset.training_images, dataset.training_labels);
  vector<LearnData> testing_data = createLearnData(dataset.test_images, dataset.test_labels);

  // Initalize Neural Network Classes

  Parameters parameters = {
      // Initial Learning Rate
      0.05,
      // Minibatch Size
      100,
      // Momentum
      0.9,
      // learnRateDecay
      0.075};

  ActivationFunction *hiddenActivation = new SigmoidActivation();
  ActivationFunction *softmax = new SoftmaxActivation();

  int image_size = 28;
  int input_nodes = image_size * image_size;
  int output_nodes = 10;

  vector<int> layerSizes = {input_nodes, 100, output_nodes};

  Network *network = new Network(layerSizes, hiddenActivation, softmax);

  int numberOfTrainingBatches = (dataset.training_images.size() / parameters.minibatchSize) - 2;

  int epochIndex = 0;
  int miniBatchIndex = 0;

  double currentLearningRate = parameters.initialLearningRate;

  double sumCorrectInTrainingEpoch = 0;

  int testingImageIndex = 1;

  //
  // SFML Windows Configuration
  //

  util::Platform platform;
  sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Window", sf::Style::Default);

  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;  // Adjust as needed

  window.create(sf::VideoMode::getFullscreenModes()[0], "SFML Fullscreen Window", sf::Style::Fullscreen, settings);
  window.setKeyRepeatEnabled(false);

  platform.setIcon(window.getSystemHandle());

  sf::Event event;

  sf::Font arial;
  arial.loadFromFile("Resources/fonts/arial.ttf");

  sf::Text settings_label;

  string epoch_label;
  string mini_batch_label;
  string accuracy_label;
  string loss_label;
  string learning_rate_label;
  string test_accuracy_label;

  epoch_label = "Epoch: " + std::to_string(epochIndex);
  mini_batch_label = "Mini Batch Index: " + std::to_string(miniBatchIndex);
  accuracy_label = "Training Accuracy:" + std::to_string(0.0);
  loss_label = "Loss:" + std::to_string(0.0);
  learning_rate_label = "Learning Rate:" + std::to_string(currentLearningRate);
  test_accuracy_label = "Test Accuracy:" + std::to_string(0.0);

  settings_label.setString(epoch_label + "\n" + mini_batch_label + "\n" + accuracy_label + "\n" + loss_label + "\n" + learning_rate_label + "\n" + test_accuracy_label);

  settings_label.setFont(arial);
  settings_label.setCharacterSize(25);
  sf::FloatRect settingsTextRect = settings_label.getLocalBounds();
  settings_label.setOrigin(settingsTextRect.left + settingsTextRect.width / 2.0f,
                           settingsTextRect.top);
  settings_label.setPosition(window.getView().getCenter().x, 0);

  State state = State::MENU;

  while (window.isOpen())
  {
    while (window.pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
        window.close();

      if (event.type == sf::Event::KeyPressed)
      {
        if (event.key.code == sf::Keyboard::Escape)
        {
          if (state == State::MENU)
          {
            window.close();
          }
          else
          {
            state = State::MENU;
          }
        }

        if (state == State::TRAINING)
        {
          if (event.key.code == sf::Keyboard::S)
          {
            cout << "Saving Network\n";
            saveNetwork(network, "network.json");
            state = State::TESTING;
          }
        }

        if (state == State::TESTING)
        {
          if (event.key.code == sf::Keyboard::Left)
          {
            testingImageIndex -= 1;
            if (testingImageIndex < 0) testingImageIndex = dataset.training_images.size() - 1;
          }

          if (event.key.code == sf::Keyboard::Right)
          {
            testingImageIndex += 1;
            if (testingImageIndex > dataset.training_images.size()) testingImageIndex = 0;
          }
        }

        if (state == State::MENU)
        {
          if (event.key.code == sf::Keyboard::T)
          {
            state = State::TRAINING;
          }

          if (event.key.code == sf::Keyboard::L)
          {
            cout << "Loading Network\n";
            network = loadNetwork("network.json");
            state = State::TESTING;
          }
        }
      }
    }
    window.clear();

    if (state == State::MENU)
    {
      string string_label = "Press \'T\' to Train new Network\nPress \'L\' to Load Network\nPress \'esc\' to Quit\n";
      settings_label.setString("Label: " + string_label);
      window.draw(settings_label);
    }
    else if (state == State::TRAINING)
    {
      int dataStartIndex = miniBatchIndex * parameters.minibatchSize;
      int dataEndIndex = (miniBatchIndex + 1) * parameters.minibatchSize;

      vector<LearnData> data_batch = slice(training_data, dataStartIndex, dataEndIndex);

      TrainResponse trainResponse = network->train(data_batch, currentLearningRate, parameters.momentum);

      double num_correct = trainResponse.num_correct;
      double cost = trainResponse.cost;

      loss_label = "Loss:" + std::to_string(cost);

      sumCorrectInTrainingEpoch += num_correct;

      if (miniBatchIndex >= numberOfTrainingBatches)
      {
        miniBatchIndex = 0;

        epochIndex++;
        epoch_label = "Epoch: " + std::to_string(epochIndex);

        sumCorrectInTrainingEpoch = 0;
        currentLearningRate = parameters.initialLearningRate / (1.0 + epochIndex * parameters.learnRateDecay);
        learning_rate_label = "Learning Rate:" + std::to_string(currentLearningRate);

        double testAccuracy = network->test(testing_data);

        test_accuracy_label = "Test Accuracy:" + std::to_string(testAccuracy);

        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

        std::mt19937 eng1(seed);

        cout << "Shuffling Training Data\n";
        std::shuffle(begin(training_data), end(training_data), eng1);
      }
      else
      {
        miniBatchIndex++;
        mini_batch_label = "Mini Batch: " + std::to_string(miniBatchIndex) + "/" + std::to_string(numberOfTrainingBatches);
      }

      accuracy_label = "Epoch Training Accuracy:" + std::to_string((double)sumCorrectInTrainingEpoch / (parameters.minibatchSize * miniBatchIndex));

      string mini_batch_accuracy = "Mini Batch Accuracy:" + std::to_string((double)num_correct / parameters.minibatchSize);

      settings_label.setString(epoch_label + "\n" + mini_batch_label + "\n" + mini_batch_accuracy + "\n" + loss_label + "\n" + accuracy_label + "\n" + learning_rate_label + "\n" + test_accuracy_label);

      window.draw(settings_label);

      // renderNodes(window, network);
    }
    else if (state == State::TESTING)
    {
      renderImage(window, training_data[testingImageIndex].image, image_size);

      string image_label = "Label: " + std::to_string(training_data[testingImageIndex].label);

      ClassifyResponse response = network->classify(training_data[testingImageIndex].image);

      double predication = response.predicted;
      string prediction_label = "Network Prediction: " + std::to_string(predication);

      vector<double> outputs = response.outputs;
      string outputs_label = "Outputs: \n";

      // aka loss
      double cost = getCost(response.outputs, generateExpectedValues(training_data[testingImageIndex].label, 10));

      string loss_label = "Loss: " + std::to_string(cost);

      for (int i = 0; i < outputs.size(); i++)
      {
        outputs_label += std::to_string(i) + ": " + std::to_string(outputs[i]) + " \n";
      }

      settings_label.setString(image_label + "\n" + prediction_label + "\n\n" + loss_label + "\n\n" + outputs_label);

      window.draw(settings_label);
    }

    window.display();
  }

  return 0;
}
