#include <stdio.h> 
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// TODO: Saving the Model

#define RICHNN_IMPLEMENTATION
#include "../RichNN.h"

void zero_gradients(NeuralNetwork* gradients) {
    for (int l = 1; l < gradients->layer_count; ++l) {
        Tensor_fill(gradients->layers[l]->weights, 0.0f);
        Tensor_fill(gradients->layers[l]->bias, 0.0f);
    }
}

int main(int argc, char** argv) 
{
    srand((unsigned)time(NULL));

    int layer_sizes[] = {2, 4, 1};
    int layer_count = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    int individual_example_inputs = 2;

    float inputs[][2] = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    float outputs[][1] = {
        {0}, {1}, {1}, {0}
    };

    int train_count = 4;

    NeuralNetwork* XorNetwork = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    init_network(XorNetwork, layer_sizes, layer_count, individual_example_inputs, train_count);

    NeuralNetwork* gradients = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    init_network(gradients, layer_sizes, layer_count, individual_example_inputs, train_count);

    float rate = 0.5f;
    int iterations = 10000;

    for (int iter = 0; iter < iterations; ++iter) 
    {
        // 1. Zero gradients
        zero_gradients(gradients);

        // 2. Accumulate gradients for each training example
        for (int j = 0; j < train_count; ++j) 
        {
            float single_input[2] = { inputs[j][0], inputs[j][1] };
            float single_output[1] = { outputs[j][0] };

            // Forward pass
            Tensor2D* output = forward(XorNetwork, single_input, individual_example_inputs, sigmoid);

            // Backprop (accumulates into gradients)
            Tensor2D* delta_L = backpropagation(XorNetwork, gradients, &single_input, single_output);

        }

        // 4. Update network with averaged gradients
        batch_gradient_descent(XorNetwork, gradients, rate);

        // Logging training loss
        if (argc > 1 && strcmp(argv[1], "-Log") == 0 && iter % 100 == 0) 
        {
            float current_cost = cost(XorNetwork, individual_example_inputs, inputs, outputs[0]);
            printf("Epoch %d / %d | Cost = %f\n", iter, iterations, current_cost);
        }
    }

    // Final loss
    float final_cost = cost(XorNetwork, individual_example_inputs, inputs, outputs[0]);
    printf("Final Loss = %f\n", final_cost);

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

    return 0;
}


