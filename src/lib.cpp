#include <vector>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;

#include "../include/lib.h"
#include "../include/learn.h"

#define M_PI 3.14159265358979323846

vector<LearnData> mutateData(vector<LearnData> data)
{
    return data;
}

vector<LearnData> slice(vector<LearnData> &arr, int X, int Y)
{
    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y + 1;

    // To store the sliced vector
    vector<LearnData> result(Y - X + 1);

    // Copy vector using copy function()
    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}
double RandomInNormalDistribution(double mean, double standardDeviation)
{
    double x1 = 1 - ((double)rand() / (RAND_MAX));
    double x2 = 1 - ((double)rand() / (RAND_MAX));

    double y1 = sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * x2);
    return y1 * standardDeviation + mean;
}

void printImage(vector<double> image)
{
    cout << "Input: \n";
    for (unsigned i = 0; i < image.size(); i++)
    {
        double c = image[i];

        if (i % 28 == 0)
        {
            cout << "\n";
        }

        cout << c << " ";
    }
    cout << "\n";
}

vector<double> generateExpectedValues(double label, int outputSize)
{
    vector<double> expectedOutputs = {};

    for (int i = 0; i < outputSize; i++)
    {
        if (i == label)
        {
            expectedOutputs.push_back(1);
        }
        else
        {
            expectedOutputs.push_back(0);
        }
    }

    return expectedOutputs;
}
vector<double> formatImage(vector<uint8_t> image)
{
    vector<double> formattedImage = {};

    for (unsigned int i = 0; i < image.size(); i++)
    {
        formattedImage.push_back((double)image[i] / 255);
    }

    return formattedImage;
}
#include <vector>

using namespace std;

double getCost(vector<double> predictedValues, vector<double> expectedValues)
{
    double cost = 0;

    int n = predictedValues.size();

    if (predictedValues.size() != expectedValues.size())
    {
        cout << predictedValues.size() << " " << expectedValues.size() << "\n";
        throw "Predicted Values length does not match Expected Values length";
    }

    for (int i = 0; i < n; i++)
    {
        double x = predictedValues[i];
        double y = expectedValues[i];

        double v = (y == 1) ? -log(x) : -log(1 - x);
        cost += isnan(v) ? 0 : v;
    }

    return cost;
}

double getCostDerivative(double predictedValue, double expectedValue)
{
    double x = predictedValue;
    double y = expectedValue;

    if (x == 0 || x == 1)
    {
        return 0;
    }

    return (-x + y) / (x * (x - 1));
}