#ifndef _RICHNN_H_
#define _RICHNN_H_

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
  int sample_size;
  int train_size;
} NeuralNetwork;

// Activation Functions 

float sigmoid(float x); 
float ReLu(float x);
float tanH(float x);

// Derivatives of Activation Functions
float dsigmoid(float dx);
float dReLu(float dx);
float dtanH(float dx);

// Creating the Network Architecture

Layer* init_layer(int index, int size, int prev_size);
void init_network(NeuralNetwork* network, int* layer_sizes, int num_layers, int sample_size, int train_size);
void print_network(NeuralNetwork* network);

// Agorithms
Tensor2D* forward(NeuralNetwork* network, float inputs[], int inputSize, float (*activationFunc)(float)); 

NeuralNetwork* batch_gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate);

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, 
    float eps, float inputs[][model->sample_size], float outputs[]);

Tensor2D* backpropagation(NeuralNetwork* model, NeuralNetwork* gradients, 
    float inputs[][model->sample_size], float outputs[]); 

float cost(NeuralNetwork* network, int n, float (*input_vals)[n], float output_vals[]); 
// Cleaning up memory

void free_layer(Layer *layer);
void free_network(NeuralNetwork *network); 

#endif // _RICHNN_H_

#ifdef RICHNN_IMPLEMENTATION

// Activation Functions

float sigmoid(float x) 
{
  return 1.f / (1.f + expf(-x));
}

float ReLu(float x) 
{
  return MAX(0.0f, x); 
}

float tanH(float x)
{
  return tanh(x);
}

// Derivatives of Activation Functions

float dsigmoid(float dx)
{
  return sigmoid(dx)*(1-sigmoid(dx));
}

float dReLu(float dx) 
{
  if (dx < 0) return 0.0f;
  else return 1.0f;
}

float dtanH(float dx) 
{
  return 1 - powf(tanh(dx), 2.0f); 
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

void init_network(NeuralNetwork* network, int* layer_sizes, int num_layers, int sample_size, int train_size) 
{
  network->layer_count = num_layers;
  network->sample_size = sample_size;
  network->train_size  = train_size;

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

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, 
                            float eps, float inputs[][model->sample_size], float outputs[]) 
{
  NeuralNetwork* m = model;

  float current_cost = cost(model, model->sample_size, inputs, outputs);
  float saved;

  for (int i = 0; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = gradients->layers[i];

    saved = current_layer->bias;
    current_layer->bias += eps;
    current_gradient_layer->bias = (cost(model, model->sample_size, inputs, outputs) - current_cost)/eps;
    current_layer->bias = saved;

    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k) 
      {
        saved = TENSOR_AT(current_layer->weights, j, k);
        TENSOR_AT(current_layer->weights, j, k) += eps;
        TENSOR_AT(current_gradient_layer->weights, j, k) = (cost(model, model->sample_size, inputs, outputs) - current_cost)/eps;
        TENSOR_AT(current_layer->weights, j, k) = saved;   
      }
    }
  }
  return gradients;
}

// backpropagation

Tensor2D* backpropagation(NeuralNetwork* model, NeuralNetwork* gradients, 
    float inputs[][model->sample_size], float outputs[]) 
{

  Tensor2D* derivatives = Tensor_init(gradients->layers[0]->weights->cols, gradients->layers[0]->weights->rows);
  int n = 2;

  int output_size = sizeof(outputs)/sizeof(outputs[0]);
  // fill the gradient matrix
  for (int i = 0; i < gradients->layer_count; ++i) 
  {
    Layer* l = gradients->layers[i];
    Tensor_fill(l->weights, 0.0f);
  }

  // loop over the inputs 
  for (int i = 0; i < model->train_size; ++i) 
  {
    float input_sample[n];
    for (int j = 0; j < model->sample_size; ++j) 
    {
      input_sample[j] = inputs[i][j];
    }
    // forward the network 
    Tensor2D* out = forward(model, (float*)input_sample, n, sigmoid);
    // compute the error for each sample 
    for (int j = 0; j < output_size; ++j) 
    {
      float error = TENSOR_AT(out, 0, j) - outputs[j];
      TENSOR_AT(derivatives, 0, j) = error;
    }
  }
#if 1
    // propagate backwards
    for (size_t l = model->layer_count - 1; l > 0; --l)
    {
      // loop over the activations 
      for (int a = 0; a < model->layers[l]->weights->cols; ++a)
      {
        float act = TENSOR_AT(model->layers[l]->weights, 0, a);
        float dact = TENSOR_AT(derivatives, 0, a);
        Layer* prev_layer_a = model->layers[l-1];
        prev_layer_a->bias += 2* (dact)*act*(1 - act);

       for (int k = 0; k < model->layers[l-1]->weights->cols; ++k)
        {
          float pa = TENSOR_AT(model->layers[l-1]->weights, 0, k);
          float w = TENSOR_AT(model->layers[l-1]->weights, k, a);
          TENSOR_AT(gradients->layers[l-1]->weights, k, a) += 2 * dact * act * (1 - act) * pa;
          TENSOR_AT(model->layers[l-1]->weights, k, a) += 2 * dact * act * (1 - act) * w;
        }

      }
    }
#endif 
  for (size_t i = 0; i < gradients->layer_count; ++i)
  {
    for (size_t j = 0; j < gradients->layers[i]->weights->rows; ++j)
    {
      for (size_t k = 0; k < gradients->layers[i]->weights->cols; ++k)
      {
        TENSOR_AT(gradients->layers[i]->weights, j, k) /= model->sample_size;
      }
    }
    
    gradients->layers[i]->bias /= model->sample_size;
  }

  Tensor_print(gradients->layers[0]->weights);


  return gradients;
} 


// computing the cost function

float cost(NeuralNetwork* network, int n, float (*input_vals)[n], float output_vals[]) 
{
  float MSE_Result = 0.0f;

  for (size_t i = 0; i < network->train_size; ++i) 
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
  MSE_Result /= network->train_size;
  return MSE_Result;
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


#endif // RICHNN_IMPLEMENTATION
