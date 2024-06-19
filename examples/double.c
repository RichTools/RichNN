#include <stdio.h> 
#include <time.h>
#include <math.h>
#include <stdlib.h>

// Perceptron Example

#define train_count (sizeof train / sizeof train[0])

float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
};

float rand_float()
{
  return (float)rand()/(float)(RAND_MAX/1);
}


float cost(float w, float b)
{
  float MSE_result = 0.0f;

  for (size_t i = 0; i < train_count; ++i)
  {
    float x = train[i][0];
    float y = (x * w) + b;
    //printf("y = %f\n", y);
    float d = y - train[i][1];
    // printf("actual: %f, expected: %f\n", y, train[i][1]);
    MSE_result += (d*d);
  }
  MSE_result /= train_count;
  return MSE_result;
}

int main() 
{
  srand(69);
  float w = rand_float()*10.0f;
  float b = rand_float()*5.0f;

  float eps = 1e-7;
  float rate = 1e-2;

  printf("%f\n", cost(w, b));

  	// we repeat this process for 500 iterations until 
	// drive the cost to 0. Again alter until correct.
	for (size_t i = 0; i < 500; ++i) {
    float c = cost(w,b);
		float dw = (cost(w + eps, b) - c)/eps;
    float db = (cost(w, b + eps) - c)/eps;
		
		w -= (rate*dw);
    b -= (rate*db);
    
    
		printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
	}

	printf("---------------\n");
	printf("w = %f, b = %f\n", w, b);

  printf("-------------------\n");
  for (int i = 0; i < train_count; i++) {
    printf("%f x 2 = %f\n", train[i][0], train[i][0]*w);
  }

  printf("\nError = %f \n", cost(w,b));
}
