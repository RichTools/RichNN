#ifndef _RICHNN_H_
#define _RICHNN_H_

#define MATRIX_IMPLEMENTATION
#include "Matrix.h"

typedef struct {
  Tensor2D* weights;
  Tensor2D* bias;
  size_t size;
} Layer;

typedef struct {
  size_t layer_count;
  Layer** layers;

  Tensor2D** activations;  
  Tensor2D** zs;          
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
// Tensor sigmoid utility
Tensor2D* Tensor_sigmoid_prime(Tensor2D* z);

// Creating the Network Architecture

Layer* init_layer(int n_out, int n_in);
void init_network(NeuralNetwork* network, int* layer_sizes, int num_layers, int sample_size, int train_size);
void print_network(NeuralNetwork* network);

// Agorithms
Tensor2D* forward(NeuralNetwork* network, float inputs[], int inputSize, float (*activationFunc)(float)); 

NeuralNetwork* batch_gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate);

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, 
    float eps, float inputs[][model->sample_size], int out, float outputs[out]);

Tensor2D* backpropagation(NeuralNetwork* model, NeuralNetwork* gradients, 
    float inputs[][model->sample_size], float outputs[]); 

float cost(NeuralNetwork* network, int n, int out, float (*input_vals)[n], float output_vals[][out]); 
//float cost(NeuralNetwork* network, int n, float (*input_vals)[n], float output_vals[]); 
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

Tensor2D* Tensor_sigmoid_prime(Tensor2D* z)
{
    Tensor2D* out = Tensor_copy(z);           // copy z shape
    for (int i = 0; i < z->rows; i++)
    {
        for (int j = 0; j < z->cols; j++)
        {
            TENSOR_AT(out, i, j) = dsigmoid(TENSOR_AT(z, i, j));
        }
    }
    return out;
}

// Neural Network Creation

Layer* init_layer(int n_out, int n_in)
{
  Layer* layer = malloc(sizeof(Layer));

  layer->size = n_out;
  layer->weights = Tensor_init(n_out, n_in); // rows = n_out, cols = n_in
  layer->bias    = Tensor_init(n_out, 1);

  // Random init weights in [-1,1) (scale as you prefer)
  for (size_t r = 0; r < layer->weights->rows; ++r) {
      for (size_t c = 0; c < layer->weights->cols; ++c) {
          TENSOR_AT(layer->weights, r, c) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
      }
  }
  for (size_t r = 0; r < layer->bias->rows; ++r) {
      TENSOR_AT(layer->bias, r, 0) = 0.0f; // or small random
  }

  return layer;
 }

void init_network(NeuralNetwork* nn,
                  int* layer_sizes,
                  int layer_count,
                  int sample_size,
                  int train_size)
{
    nn->layer_count = layer_count;
    nn->sample_size = sample_size;
    nn->train_size = train_size;

    nn->layers = malloc(layer_count * sizeof(Layer*));
    nn->activations = malloc(layer_count * sizeof(Tensor2D*));
    nn->zs = malloc(layer_count * sizeof(Tensor2D*));

    for (int l = 0; l < layer_count; ++l) {
        nn->layers[l] = NULL;
        nn->activations[l] = NULL;
        nn->zs[l] = NULL;
    }

    // Note: layer 0 has no weights/bias
    //nn->layers[0] = NULL;

    for (int l = 0; l < layer_count; ++l) {
        int n_out = layer_sizes[l];
        int n_in  = layer_sizes[l-1];

        nn->layers[l] = init_layer(n_out, n_in);
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
      Tensor_print(network->layers[i]->bias);
    }
    printf("\n\n");
  }
}

// Forward Propagation through the network

Tensor2D* forward(NeuralNetwork* network,
                  float inputs[],
                  int inputSize,
                  float (*activationFunc)(float))
{
    int L = (int)network->layer_count;

    // build input activation vector (activations[0])
    if (network->activations[0]) {
        // reuse existing tensor (optional); otherwise re-init
        Tensor_free(network->activations[0]);
    }
    network->activations[0] = Tensor_init(inputSize, 1);
    for (int i = 0; i < inputSize; ++i) {
        TENSOR_AT(network->activations[0], i, 0) = inputs[i];
    }

    Tensor2D* a_prev = network->activations[0];

    // Forward through layers 1..L-1
    for (int l = 1; l < L; ++l) {
        Layer* layer = network->layers[l];

        // z = W * a_prev + b
        Tensor2D* prod = Tensor_mul(layer->weights, a_prev); // shape n_out x 1
        if (network->zs[l]) Tensor_free(network->zs[l]);
        network->zs[l] = Tensor_add(prod, layer->bias);
        Tensor_free(prod);

        // compute activation a = sigma(z)
        if (network->activations[l]) Tensor_free(network->activations[l]);
        network->activations[l] = Tensor_copy(network->zs[l]); // copy shape n_out x 1
        if (l == L - 1) {
            Tensor_map(network->activations[l], sigmoid); // final activation sigmoid
        } else {
            Tensor_map(network->activations[l], activationFunc);
        }

        a_prev = network->activations[l];
    }

    return network->activations[L-1]; // note: last index is L-1
}


NeuralNetwork* batch_gradient_descent(NeuralNetwork* model, NeuralNetwork* gradients, float rate) 
{
  NeuralNetwork* m = model;
  NeuralNetwork* g = gradients;

  for (int i = 1; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = g->layers[i];

    for (int r = 0; r < current_layer->bias->rows; r++)
    {
        TENSOR_AT(current_layer->bias, r, 0) -=
            rate * TENSOR_AT(current_gradient_layer->bias, r, 0) / m->train_size;
    }


    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k)
      {
        TENSOR_AT(current_layer->weights, j, k) -= (rate * TENSOR_AT(current_gradient_layer->weights, j, k)) / m->train_size;
      }
    }
  }
  return m;
}

Tensor2D* finite_difference(NeuralNetwork* model, NeuralNetwork* gradients, 
                            float eps, float inputs[][model->sample_size], int out, float outputs[out]) 
{
  NeuralNetwork* m = model;

  float current_cost = cost(model, model->sample_size, out, inputs, outputs);
  float saved;

  for (int i = 0; i < m->layer_count; ++i)
  {
    Layer* current_layer = m->layers[i];
    Layer* current_gradient_layer = gradients->layers[i];

    for (int r = 0; r < current_layer->bias->rows; r++) {
      float saved = TENSOR_AT(current_layer->bias, r, 0);
      TENSOR_AT(current_layer->bias, r, 0) += eps;

      TENSOR_AT(current_gradient_layer->bias, r, 0) =
          (cost(model, model->sample_size, out, inputs, outputs) - current_cost) / eps;

      TENSOR_AT(current_layer->bias, r, 0) = saved;
    }

    for (int j = 0; j < current_layer->weights->rows; ++j) 
    {
      for (int k = 0; k < current_layer->weights->cols; ++k) 
      {
        saved = TENSOR_AT(current_layer->weights, j, k);
        TENSOR_AT(current_layer->weights, j, k) += eps;
        TENSOR_AT(current_gradient_layer->weights, j, k) = (cost(model, model->sample_size, out, inputs, outputs) - current_cost)/eps;
        TENSOR_AT(current_layer->weights, j, k) = saved;   
      }
    }
  }
  return gradients;
}

// backpropagation
// http://neuralnetworksanddeeplearning.com/chap2.html

Tensor2D* backpropagation(NeuralNetwork* model, NeuralNetwork* gradients, 
    float inputs[][model->sample_size], float outputs[]) 
{
  int L = model->layer_count;
  int output_size = model->layers[L - 1]->size;

  // 1. Compute δ_L = (a_L – y) ⊙ σ′(z_L)

  Tensor2D* aL = model->activations[L - 1];
  Tensor2D* zL = model->zs[L - 1];

  // build y vector 
  Tensor2D* y = Tensor_init(output_size, 1);
  for (int i = 0; i < output_size; i++)
    TENSOR_AT(y, i, 0) = outputs[i];

  Tensor2D* diff = Tensor_sub(aL, y); // (a_L - y)
                                      
  Tensor2D* sigpL = Tensor_sigmoid_prime(zL);      // σ′(z_L)
  Tensor2D* delta = Tensor_hadamard(diff, sigpL);  // δ_L
  
  Tensor2D* d_final = Tensor_copy(delta); // store δ_L to returning

  // Store grads for output layer 
  Tensor2D* a_prev = model->activations[L - 2];
  Tensor2D* a_prev_T = Tensor_transpose(a_prev);

  Tensor2D* dW_L = Tensor_mul(delta, a_prev_T); // ∇W_L
                                                

  for (int i = 0; i < gradients->layers[L-1]->weights->rows; i++) 
  {
    for (int j = 0; j < gradients->layers[L-1]->weights->cols; j++)
    {
      TENSOR_AT(gradients->layers[L-1]->weights, i, j) += 
        TENSOR_AT(dW_L, i, j);
    }
  }
  for (int i = 0; i < gradients->layers[L-1]->bias->rows; i++)
  {
    TENSOR_AT(gradients->layers[L-1]->bias, i, 0) += 
      TENSOR_AT(delta, i, 0);
  }
  
  // 2. Backpropagate through hidden layers
  // δ_l = (W_(l+1)^T δ_(l+1)) ⊙ σ′(z_l)
  
  Tensor2D* delta_next = delta;
  
  for (int l = L - 2; l >= 1; l--) 
  {
    Tensor2D* W_next = model->layers[l + 1]->weights;
    Tensor2D* W_next_T = Tensor_transpose(W_next);
    
    Tensor2D* tmp = Tensor_mul(W_next_T, delta_next); // W^T δ_(l+1)
    Tensor2D* sigp = Tensor_sigmoid_prime(model->zs[l]);

    Tensor2D* delta_l = Tensor_hadamard(tmp, sigp);

    // ∇W_l = δ_l * a_(l-1)^T
    Tensor2D* a_prev_l_T = Tensor_transpose(model->activations[l - 1]);
    Tensor2D* dW_l = Tensor_mul(delta_l, a_prev_l_T);


    for (int i = 0; i < gradients->layers[l]->weights->rows; i++)
    {
      for (int j = 0; j < gradients->layers[l]->weights->cols; j++)
      {
        TENSOR_AT(gradients->layers[l]->weights, i, j) +=
          TENSOR_AT(dW_l, i, j);
      }
    }
    for (int i = 0; i < gradients->layers[l]->bias->rows; i++)
    {
      TENSOR_AT(gradients->layers[l]->bias, i, 0) +=
        TENSOR_AT(delta_l, i, 0);
    }

    Tensor_free(W_next_T);
    Tensor_free(tmp);
    Tensor_free(sigp);
    Tensor_free(a_prev_l_T);
    Tensor_free(dW_l);

    if (l != L - 2) Tensor_free(delta_next);

    delta_next = delta_l;
  }

  Tensor_free(y);
  Tensor_free(diff);
  Tensor_free(sigpL);
  Tensor_free(delta);
  Tensor_free(a_prev_T);
  Tensor_free(dW_L);

  return d_final;
} 


// computing the cost function

float cost(NeuralNetwork* network, int n, int out, float (*input_vals)[n], float output_vals[][out]) 
{
  float MSE = 0.0f;

  int N = network->train_size;
  int output_size = network->layers[network->layer_count - 1]->size;

  for (int i = 0; i < N; i++)
  {
    // Copy input
    float input_sample[n];
    for (int j = 0; j < n; j++)
      input_sample[j] = input_vals[i][j];

    // Forward pass
    Tensor2D* pred = forward(network, input_sample, n, sigmoid);

    // MSE across all outputs 
    for (int k = 0; k < output_size; k++)
    {
      float y = TENSOR_AT(pred, k, 0);
      float t = output_vals[i][k];
      float d = y - t;
      MSE += d * d;
    }
  }

  // Normalize by total samples * output dimensions
  MSE /= (float)(N * output_size);

  return MSE;
}

// Free allocated memory for layers
void free_layer(Layer *layer) 
{
  Tensor_free(layer->weights);
  Tensor_free(layer->bias);
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
