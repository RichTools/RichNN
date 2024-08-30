#include <stdio.h>
#include "Matrix.h"
#include "Vector.h"

typedef struct {
  Tensor2D* weights;
  Vector* bias;
  size_t input_size;
  size_t hidden_size;
} Layer;

typedef struct {
  Layer* forget_gate;
  Layer* input_gate;
  Layer* candidate_gate;
  Layer* output_gate;
  Layer* final_gate;
} LSTMCell;

typedef struct {
  LSTMCell** cells;
  size_t num_layers;
  size_t input_size;
  size_t hidden_size;
} LSTM;

