# Nueral Network

## Background

Handwritten number classification trained on the [MNIST DATASET](https://en.wikipedia.org/wiki/MNIST_database)

A nueral network, built from scratch in c+.
Houses 3 layers, with 784 inputs, 128 hidden layer, and 10 output nuerons respectively.

Test sample classification rate of 80%.

## Tools and Software

Built with c++, on windows 11, with the GNU compiler (g++).
Uses the c++ MNIST library for reading images and labels from database.

## Test Samples


## Save/Load Functionality

Saves data to ```network.txt```, of data partaining:

    - number of layers
    - nuerons per layer
    - weights per connection
    - biases per nueron

Which can be loaded in after training time to apply the trained model to classify external sources.