#ifndef _RICHML_H_
#define _RICHML_H_

#define MATRIX_IMPLEMENTATION
#include "Matrix.h"

typedef struct {
  Tensor2D* weights;
  float bias;
  size_t size;
} Layer;

typedef struct {
  size_t layer_count;
  Layer** layers;
} NeuralNetwork;

// Activation Functions 

float sigmoid(float x); 
float ReLu(float x);

// Creating the Network Architecture

Layer* init_layer(int index, int size, int prev_size);
void init_network(NeuralNetwork* network, int* layer_sizes, int num_layers);
void print_network(NeuralNetwork* network);

// Agorithms
Tensor2D* forward(NeuralNetwork* network, float inputs[], int inputSize, float (*activationFunc)(float)); 
NeuralNetwork* batch_gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate);

  // Cleaning up memory

void free_layer(Layer *layer);
void free_network(NeuralNetwork *network); 

#endif // _RICHML_H_

#ifdef RICHML_IMPLEMENTATION

// Activation Functions

float sigmoid(float x) 
{
  return 1.f / (1.f + expf(-x));
}

float ReLu(float x) 
{
  return MAX(0.0f, x); 
}

// Neural Network Creation

Layer* init_layer(int index, int size, int prev_size)
{
  Layer* layer = (Layer*)malloc(sizeof(Layer*));

  layer->size = size;
 
  layer->weights = Tensor_init(size, prev_size);
  if (index == 0)
  {
    layer->bias = 0; 
  } 
  else
  {
    layer->bias = rand_float();
  }
  return layer;
 }

void init_network(NeuralNetwork* network, int* layer_sizes, int num_layers) 
{
  network->layer_count = num_layers;

  network->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);

  for (int l = 0; l < num_layers; ++l)
  {
    int prev_size = (l == 0) ? 1 : layer_sizes[l - 1];
    network->layers[l] = init_layer(l, layer_sizes[l], prev_size);
  }
}

void print_network(NeuralNetwork* network)
{
  for (int i = 0; i < network->layer_count; ++i)
  {
    printf("====  Layer %d ==== \n", i);
    if (i == 0)
      printf("Inputs: \n");
    else
      printf("Weights: \n");
    Tensor_print(network->layers[i]->weights);
    if (i != 0)
    {
      printf("Bias: \n");
      printf("%f", network->layers[i]->bias);
    }
    printf("\n\n");
  }
}

// Forward Propagation through the network

Tensor2D* forward(NeuralNetwork* network, float inputs[], int inputSize, float (*activationFunc)(float)) 
{
  // put the inputs on the input layer. May need to move else where
  for (int in = 0; in < inputSize; ++in)
  {
    TENSOR_AT(network->layers[0]->weights, in, 0) = inputs[in];
  }

  // get the input layer
  Tensor2D* layerInput = network->layers[0]->weights;

  for (size_t i = 0; i < network->layer_count-1; ++i)
  {
    Tensor2D* nextLayer = 
      Tensor_add(
        Tensor_mul(
          network->layers[i+1]->weights,
          layerInput
        ),
        network->layers[i]->bias
    );
    if (i == network->layer_count-1) 
    {
      // sigmoid is preferred on the last layer since it ensures 0 <= a <= 1
      Tensor_map(nextLayer, sigmoid);
    } 
    else 
    {
      // here we can use any activation we want
      Tensor_map(nextLayer, activationFunc);
    }
    layerInput = nextLayer;
  }
  return layerInput; // output layer 
}

NeuralNetwork* batch_gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate) 
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


// Free allocated memory for layers
void free_layer(Layer *layer) 
{
  Tensor_free(layer->weights);
  free(layer);
}

// Free allocated memory for the neural network
void free_network(NeuralNetwork *network) 
{
  for (int i = 0; i < network->layer_count; i++) {
      free_layer(network->layers[i]);
  }
  free(network);
}


#endif // RICHML_IMPLEMENTATION
