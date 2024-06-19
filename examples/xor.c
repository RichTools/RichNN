#include <stdio.h> 
#include <time.h>
#include <math.h>
#include <stdlib.h>

// TODO: Saving the Model
// TODO: Move the Cost Function and FiniteDifference into the header

#define RICHML_IMPLEMENTATION
#include "../RichML.h"

#define train_count (sizeof train / sizeof train[0])

// Trainining Data

float train[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

// computing the cost function

float cost(NeuralNetwork* network) 
{
  float MSE_Result = 0.0f;
  int outputIdx = 2;

  for (size_t i = 0; i < train_count; ++i) 
  {
    float inputs[] = {train[i][0], train[i][1]};

    Tensor2D* output_matrix = forward(network, (float*)inputs, 2, sigmoid);

    if (output_matrix->cols * output_matrix->rows == 1)
    {
      float y = TENSOR_AT(output_matrix, 0, 0);
      float d = y - train[i][outputIdx];
      MSE_Result += d*d;
    }
    else 
    {
      TENSOR_ASSERT(0 && "Mutiple Output Parameters are not Implemented");
    }
  }
  MSE_Result /= train_count;
  return MSE_Result;
}

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, float eps) 
{
  NeuralNetwork* m = model;

  float current_cost = cost(model);
  float saved;

  for (int i = 0; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = gradients->layers[i];

    saved = current_layer->bias;
    current_layer->bias += eps;
    current_gradient_layer->bias = (cost(model) - current_cost)/eps;
    current_layer->bias = saved;

    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k) 
      {
        saved = TENSOR_AT(current_layer->weights, j, k);
        TENSOR_AT(current_layer->weights, j, k) += eps;
        TENSOR_AT(current_gradient_layer->weights, j, k) = (cost(model) - current_cost)/eps;
        TENSOR_AT(current_layer->weights, j, k) = saved;   
      }
    }
  }
  return gradients;
}

NeuralNetwork* gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate) 
{
  NeuralNetwork* m = model;
  NeuralNetwork* g = gradients;

  for (int i = 0; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = g->layers[i];

    current_layer->bias -= (rate * current_gradient_layer->bias);

    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k)
      {
        TENSOR_AT(current_layer->weights, j, k) -= (rate * TENSOR_AT(current_gradient_layer->weights, j, k));
      }
    }
  }
  return m;
}



int main(void) 
{
  srand(time(NULL));
  
  int layer_sizes[] = {2, 2, 2, 1};
  int layer_count = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  NeuralNetwork* XorNetwork = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  init_network(XorNetwork, layer_sizes, layer_count);

  print_network(XorNetwork);
  
  NeuralNetwork* gradients = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  init_network(gradients, layer_sizes, layer_count);

  float eps = 1e-3;
  float rate = 1e-1;

  int iterations = 50 * 1000;

  for (size_t i = 0; i <= iterations; ++i) 
  {
    finite_difference(XorNetwork, gradients, eps);
    XorNetwork = gradient_descent(XorNetwork, gradients, rate);
    if (i % 100 == 0) printf("Epoch - %zu / %d) cost = %f\n", i, iterations, cost(XorNetwork));
  }

  printf("---------------\n");
  printf("Loss = %f\n", cost(XorNetwork));

  printf("---------------");
  printf("\nValidation Data: \n");

  for (size_t i = 0; i < 2; ++i) 
  {
      for (size_t j = 0; j < 2; ++j)
      {
        float inputs[] = {i, j};
        Tensor2D* output = forward(XorNetwork, (float*)inputs, 2, sigmoid);
        printf("\n");
        float o = (TENSOR_AT(output, 0,0) < 0.1) ? 0 : 1;
        printf("%zu ^ %zu = %f\n", i, j, o);
      }
  }
  
  free_network(XorNetwork);
}
