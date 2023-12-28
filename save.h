#ifndef SAVE_H
#define SAVE_H

using namespace std;

#include <fstream>

#include "network.h"
#include "save.h"

void saveNetwork(Network* network, string filename);
Network loadNetwork(string filename);

#endif