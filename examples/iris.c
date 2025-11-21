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

int argmax(Tensor2D* out) {
    int rows = out->rows;
    int max_index = 0;
    float max_val = TENSOR_AT(out, 0, 0);

    for (int i = 1; i < rows; i++) {
        float val = TENSOR_AT(out, i, 0);
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }
    return max_index;
}


#define MAX_LINE 128
#define IRIS_FEATURES 4
#define IRIS_CLASSES 3
#define IRIS_ROWS 150
#define TRAIN_SPLIT 0.8

void shuffle_data(float X[][IRIS_FEATURES], int y[], int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        // Swap X[i] and X[j]
        for (int k = 0; k < IRIS_FEATURES; k++) {
            float tmp = X[i][k];
            X[i][k] = X[j][k];
            X[j][k] = tmp;
        }

        // Swap y[i] and y[j]
        int tmp_y = y[i];
        y[i] = y[j];
        y[j] = tmp_y;
    }
}

// Convert class string to integer index
int class_to_int(const char* label)
{
    if (strcmp(label, "Iris-setosa") == 0) return 0;
    if (strcmp(label, "Iris-versicolor") == 0) return 1;
    if (strcmp(label, "Iris-virginica") == 0) return 2;
    return -1;
}

int load_iris(const char* filename, float X[IRIS_ROWS][IRIS_FEATURES], int y[IRIS_ROWS])
{
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Could not open iris.data");
        return 0;
    }

    char line[MAX_LINE];
    int i = 0;

    while (fgets(line, MAX_LINE, f) && i < IRIS_ROWS)
    {
        float f1, f2, f3, f4;
        char label[32];

        // Parse CSV
        if (sscanf(line, "%f,%f,%f,%f,%31s", &f1, &f2, &f3, &f4, label) == 5)
        {
            X[i][0] = f1;
            X[i][1] = f2;
            X[i][2] = f3;
            X[i][3] = f4;

            // Convert class label to integer
            y[i] = class_to_int(label);
            i++;
        }
    }
    fclose(f);
    return i; // number of rows loaded
}

void one_hot(int label, float out[3])
{
    out[0] = out[1] = out[2] = 0;
    out[label] = 1.0f;
}


int main() 
{
  srand((unsigned)time(NULL));
  
  float X[IRIS_ROWS][IRIS_FEATURES];
  int y[IRIS_ROWS];

  int count = load_iris("./data/iris.data", X, y);

  printf("Loaded %d rows\n", count);

  shuffle_data(X, y, count);

  int train_count = (int)(count * TRAIN_SPLIT);
  int test_count  = count - train_count;

  printf("Train: %d   Test: %d\n", train_count, test_count);

  //-----------------------------
  // 2. Network configuration
  //-----------------------------
  int layer_sizes[] = {4, 8, 3};   // input=4, hidden=8, output=3
  int layer_count = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
  init_network(net, layer_sizes, layer_count, IRIS_FEATURES, train_count);

  NeuralNetwork* gradients = malloc(sizeof(NeuralNetwork));
  init_network(gradients, layer_sizes, layer_count, IRIS_FEATURES, train_count);

  float rate = 0.05f;
  int epochs = 3000;

  //-----------------------------
  // 3. Training Loop
  //-----------------------------
  for (int e = 0; e < epochs; e++)
  {
    zero_gradients(gradients);

    for (int i = 0; i < train_count; i++)
    {
      float single_input[4] = {
        X[i][0], X[i][1], X[i][2], X[i][3]
      };

      // one-hot label
      float target[3];
      one_hot(y[i], target);

      forward(net, single_input, 4, sigmoid);
      backpropagation(net, gradients, &single_input, target);
    }

    batch_gradient_descent(net, gradients, rate);

    float Y[IRIS_ROWS][IRIS_CLASSES];
    for (int i = 0; i < train_count; ++i) {
      one_hot(y[i], Y[i]);
    }

    if (e % 100 == 0)
    {
        float c = cost(net, IRIS_FEATURES, IRIS_CLASSES, X, Y);
        printf("Epoch %d/%d | Cost = %f\n", e, epochs, c);
    }
  }

  //-----------------------------
  // 4. Test on all samples
  //-----------------------------
  printf("\nTesting classifier:\n");
  int correct = 0;

  for (int i = train_count; i < count; i++) // print first 10 examples
  {
      Tensor2D* out = forward(net, X[i], IRIS_FEATURES, sigmoid);

      printf("Input: [%f %f %f %f] -> ", 
          X[i][0], X[i][1], X[i][2], X[i][3]);

      int predicted = argmax(out);

      if (predicted == y[i]) correct++;

      printf("Predicted = %d   Actual = %d\n", predicted, y[i]);
    }

    printf("\nTEST ACCURACY = %.2f%%\n",
          100.0 * correct / test_count);

    return 0;
}



