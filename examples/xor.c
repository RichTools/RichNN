#include <stdio.h> 
#include <time.h>
#include <math.h>
#include <stdlib.h>

// TODO: Saving the Model
// TODO: Move the Cost Function and FiniteDifference into the header

#define RICHNN_IMPLEMENTATION
#include "../RichNN.h"

#define train_count (sizeof train / sizeof train[0])

// Trainining Data

float train[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};


#define n_i 2
static const float inputs[][n_i] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const float outputs[] = {0, 1, 1, 0};
#define train_count (sizeof inputs / sizeof inputs[0])

// computing the cost function

float cost(NeuralNetwork* network, int n, float (*input_vals)[n], float output_vals[]) 
{
  float MSE_Result = 0.0f;

  for (size_t i = 0; i < train_count; ++i) 
  {
    float input_sample[n];

    for (int j = 0; j < n; ++j)
    {
      input_sample[j] = input_vals[i][j];
    }

    Tensor2D* output_matrix = forward(network, (float*)input_sample, n, sigmoid);

    if (output_matrix->cols * output_matrix->rows == 1)
    {
      float y = TENSOR_AT(output_matrix, 0, 0);
      float d = y - output_vals[i];
      MSE_Result += d*d;
    }
    else 
    {
      TENSOR_ASSERT(0 && "Multiple Output Parameters are not Implemented");
    }
  }
  MSE_Result /= train_count;
  return MSE_Result;
}

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, 
                            float eps, float inputs[][3], float outputs[]) 
{
  NeuralNetwork* m = model;

  float current_cost = cost(model, n_i, inputs, outputs);
  float saved;

  for (int i = 0; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = gradients->layers[i];

    saved = current_layer->bias;
    current_layer->bias += eps;
    current_gradient_layer->bias = (cost(model, n_i, inputs, outputs) - current_cost)/eps;
    current_layer->bias = saved;

    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k) 
      {
        saved = TENSOR_AT(current_layer->weights, j, k);
        TENSOR_AT(current_layer->weights, j, k) += eps;
        TENSOR_AT(current_gradient_layer->weights, j, k) = (cost(model, n_i, inputs, outputs) - current_cost)/eps;
        TENSOR_AT(current_layer->weights, j, k) = saved;   
      }
    }
  }
  return gradients;
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
    finite_difference(XorNetwork, gradients, eps, inputs, outputs);
    XorNetwork = batch_gradient_descent(XorNetwork, gradients, rate);
    if (i % 100 == 0) printf("Epoch - %zu / %d) cost = %f\n", i, iterations, cost(XorNetwork, n_i, inputs, outputs));
  }

  printf("---------------\n");
  printf("Loss = %f\n", cost(XorNetwork, n_i, inputs, outputs));

  printf("---------------");
  printf("\nValidation Data: \n");

  for (size_t i = 0; i < 2; ++i) 
  {
      for (size_t j = 0; j < 2; ++j)
      {
        float inputs[] = {i, j};
        Tensor2D* output = forward(XorNetwork, (float*)inputs, n_i, sigmoid);
        printf("\n");
        float o = (TENSOR_AT(output, 0,0) < 0.1) ? 0 : 1;
        printf("%zu ^ %zu = %f\n", i, j, o);
      }
  }
  
  free_network(XorNetwork);
}
