
#ifndef DRAW_H
#define DRAW_H

using namespace std;

#include <vector>
#include <SFML/Graphics.hpp>
#include "network.h"

void updateLabels(sf::RenderWindow& window, sf::Text& epoch_label, sf::Text& mini_batch_label, sf::Text& accuracy_label, sf::Text& loss_label, sf::Text& learning_rate_label, sf::Text& test_accuracy_label);
void createStandardLabel(int x, int y, sf::Font font, sf::Text* label);
void renderNodes(sf::RenderWindow& window, Network* network);
void renderImage(sf::RenderWindow& window, vector<double> image, int image_size);

#endif