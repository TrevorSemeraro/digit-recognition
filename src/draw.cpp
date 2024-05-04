#include "draw.h"

#include <SFML/Graphics.hpp>
#include <vector>

#include "network.h"

void renderNodes(sf::RenderWindow& window, Network* network)
{
  int x_padding = 50;
  int y_padding = 10;
  int node_drawing_radius = 2;

  for (auto it = network->layers.begin(); it != network->layers.end(); ++it)
  {
    Layer layer = *it;
    int index = std::distance(network->layers.begin(), it);

    for (int i = 0; i < layer.nodes_in; i++)
    {
      sf::CircleShape circle(node_drawing_radius);
      // Adjust weight based on neurons bias
      double bias = layer.biases[i];
      int color_value = (int)(abs(bias) * 255);
      // int color_value = 255;
      sf::Color color(color_value, color_value, color_value);

      circle.setFillColor(color);

      circle.setPosition(node_drawing_radius * index + x_padding * index, node_drawing_radius * i + y_padding * i);
      window.draw(circle);
    }

    for (int i = 0; i < layer.nodes_in; i++)
    {
      for (int j = 0; j < layer.nodes_out; j++)
      {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(
                node_drawing_radius * index + x_padding * index + node_drawing_radius,
                node_drawing_radius * i + y_padding * i)),
            sf::Vertex(sf::Vector2f(
                node_drawing_radius * (index + 1) + x_padding * (index + 1),
                node_drawing_radius * j + y_padding * j)),
        };

        double weight = layer.weights[i][j];
        int color_value = (int)(abs(weight) * 255);
        sf::Color color(color_value, color_value, color_value);

        line[0].color = color;
        line[1].color = color;

        window.draw(line, 2, sf::Lines);
      }
    }
  }
  // Draw output nodes
  auto lastLayer = network->layers.back();

  for (int i = 0; i < lastLayer.nodes_out; i++)
  {
    sf::CircleShape circle(node_drawing_radius);
    circle.setFillColor(sf::Color::Green);
    circle.setPosition(node_drawing_radius * (network->layers.size()) + x_padding * (network->layers.size()), node_drawing_radius * i + y_padding * i);
    window.draw(circle);
  }
}

const int scaleFactor = 10;
const int lengthOfRGBA = 4;
void renderImage(sf::RenderWindow& window, vector<double> image, int image_size)
{
  int scaledImageSize = image_size * scaleFactor;
  sf::Uint8* pixels = new sf::Uint8[scaledImageSize * scaledImageSize * lengthOfRGBA];

  for (int x = 0; x < image_size; x += 1)
  {
    for (int y = 0; y < image_size; y += 1)
    {
      sf::Uint8 current_pixel_value = static_cast<unsigned int>(image[x + y * image_size] * 255);

      for (int xscale = 0; xscale < scaleFactor; xscale++)
      {
        for (int yscale = 0; yscale < scaleFactor; yscale++)
        {
          // Non-scaled image pixel index
          // int pixel_index = (x + y * image_size) * lengthOfRGBA;

          int pixel_index = (x * scaleFactor + xscale + (y * scaleFactor + yscale) * scaledImageSize) * lengthOfRGBA;

          pixels[pixel_index] = current_pixel_value;
          pixels[pixel_index + 1] = current_pixel_value;
          pixels[pixel_index + 2] = current_pixel_value;
          pixels[pixel_index + 3] = 255;
        }
      }
    }
  }

  sf::Texture texture;
  texture.create(scaledImageSize, scaledImageSize);

  texture.update(pixels);

  sf::Sprite sprite(texture);
  window.draw(sprite);

  delete[] pixels;
}