#include <stdio.h> 
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// TODO: Saving the Model

#define RICHNN_IMPLEMENTATION
#include "../RichNN.h"

// Trainining Data

static const float inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const float outputs[] = {0, 1, 1, 0};
#define train_count (sizeof inputs / sizeof inputs[0])


int main(int argc, char** argv) 
{
  srand(time(NULL));
  
  // define the number of neurons on each layer. 
  int layer_sizes[] = {2, 2, 2, 1};
  // compute the number of layers - we could store this wi
  int layer_count = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
  // here we store the number of inputs per example
  int individual_example_inputs = 2;

  // Create the neural network
  NeuralNetwork* XorNetwork = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  init_network(XorNetwork, 
               layer_sizes, layer_count, 
               individual_example_inputs, train_count
               );

  // create seperate network to store the gradients on each layer
  NeuralNetwork* gradients = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  init_network(gradients, 
               layer_sizes, layer_count, 
               individual_example_inputs, train_count
               );

  // define an epsilon and learning rate for the network
  float eps = 1e-3;
  float rate = 1e-1;

  int iterations = 50 * 1000;

  for (size_t i = 0; i <= iterations; ++i) 
  {
    // train the network for a specified number of iterations 
    finite_difference(XorNetwork, gradients, eps, inputs, outputs);
    XorNetwork = batch_gradient_descent(XorNetwork, gradients, rate);

    // this will show us logs if we provide the -Log flag (see the build script)
    if(argc > 1 && strcmp(argv[1], "-Log") == 0)
    {
      if (i % 100 == 0) 
      {
        float cost_value = cost(XorNetwork, XorNetwork->sample_size, inputs, outputs);
        printf("Epoch - %zu / %d) cost = %f\n", i, iterations, cost_value);
      }
    }
  }

  // validating that our network performed well, since we only have those inputs possible 
  // this isn't great but it shows us that it worked correctly.
  printf("---------------\n");
  printf("Loss = %f\n", cost(XorNetwork, XorNetwork->sample_size, inputs, outputs));

  printf("---------------");
  printf("\nValidation Data: \n");

  for (size_t i = 0; i < 2; ++i) 
  {
      for (size_t j = 0; j < 2; ++j)
      {
        float inputs[] = {i, j};
        Tensor2D* output = forward(XorNetwork, (float*)inputs, XorNetwork->sample_size, sigmoid);
        printf("\n");
        float o = (TENSOR_AT(output, 0,0) < 0.1) ? 0 : 1;
        printf("%zu ^ %zu = %f\n", i, j, o);
      }
  }
  
  free_network(XorNetwork);
}
