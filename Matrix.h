#ifndef _MATRIX_H_
#define _MATRIX_H_

typedef struct {
  int rows;
  int cols;
  float **data;
} Tensor2D;

/* Note to Reader:
 * I am using the terms, Tensor and Matrix interchangeably, 
 * purely because I can, and no other reason.
 */

float rand_float(void);

// Memory Management

Tensor2D* Tensor_alloc(int rows, int cols); // allocate memory for the matrix, rox x col x sizeof float
void Tensor_free(Tensor2D* matrix); // free the memory for a tensor -> the data, and the tensor itself.
Tensor2D* Tensor_init(int rows, int cols); // calls Tensor_alloc and initalised with r x c random float values

// Operations 

Tensor2D* Tensor_mul(Tensor2D* a, Tensor2D* b); // mutiply two tensors together
Tensor2D* Tensor_add(Tensor2D* a, float b); // add a scalar vlaue into a tensor
Tensor2D* Tensor_scale(Tensor2D* a, int scalar); // scale a tensor by a scalar value
void Tensor_map(Tensor2D* a, float (*func)(float)); // map a function over each value in the tensor 

// Utils 

void Tensor_print(Tensor2D* tensor);  // print the tensor into the terminal 
void Tensor_fill(Tensor2D* matrix, float val); // will the tensor will a specified val 
void Tensor_Dim(Tensor2D* t); // show the dimensions of the tensor

#ifndef TENSOR_ASSERT
#include <assert.h>
#define TENSOR_ASSERT assert
#endif // TENSOR_ASSERT

// Macro Definitions 

// get the value at location specified by row and col (x, y)
#define TENSOR_AT(t, x, y) t->data[x][y] 
// return the larger of the two passed in values
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

#endif // _MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION

float rand_float(void)
{
  return (float)rand()/(float)(RAND_MAX/1);
}

Tensor2D* Tensor_alloc(int rows, int cols)
{
  Tensor2D* matrix = (Tensor2D*)malloc(sizeof(Tensor2D));
  if (!matrix) 
  {
    fprintf(stderr, "Matrix Allocation Failed");
    exit(1);
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->data = (float**)malloc(rows * sizeof(float*));
  
  if (!matrix->data) {
      fprintf(stderr, "Memory allocation failed\n");
      free(matrix);
      exit(1);
  }

  for (int i = 0; i < rows; ++i) 
  {
    matrix->data[i] = (float *)malloc(cols * sizeof(float));
    if (!matrix->data[i]) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        for (int j = 0; j < i; ++j) 
        {
            free(matrix->data[j]);
        }
        
        free(matrix->data);
        free(matrix);
        exit(1);
    }
  }
  return matrix;
}

Tensor2D* Tensor_init(int rows, int cols) 
{
  Tensor2D* tensor = Tensor_alloc(rows, cols);

  for (int i = 0; i < rows; ++i) 
  {
    for (int j = 0; j < cols; ++j)
    {
      tensor->data[i][j] = rand_float(); 
    }
  }

  return tensor;
}

Tensor2D* Tensor_scale(Tensor2D* a, int scalar) 
{
  Tensor2D* result = Tensor_alloc(a->rows, a->cols);

  for (int i = 0; i < a->rows; ++i)
  {
    for (int j = 0; j < a->cols; ++j) 
    {
      TENSOR_AT(result, i, j) = TENSOR_AT(a, i, j) * scalar;
    }
  }
  return result;
}

void Tensor_fill(Tensor2D* matrix, float val) 
{
  Tensor2D* result = Tensor_alloc(matrix->rows, matrix->cols);

  for (int i = 0; i < result->rows; ++i) 
  {
    for (int j = 0; j < result->cols; ++j)
    {
      TENSOR_AT(matrix, i, j) = val;
    }
  }
}


Tensor2D* Tensor_add(Tensor2D* a, float b)
{
  Tensor2D* result = Tensor_alloc(a->rows, a->cols);

  for (int i = 0; i < result->rows; ++i) 
  {
    for (int j = 0; j < result->cols; ++j)
    {
      TENSOR_AT(result, i, j) = TENSOR_AT(a, i, j) + b;
    }
  }
  return result;
}

Tensor2D* Tensor_mul(Tensor2D* a, Tensor2D* b) 
{
  // TODO: Strassen's Algorithm?
  TENSOR_ASSERT(a->cols == b->rows);
  Tensor2D* result = Tensor_alloc(a->rows, b->cols);
  TENSOR_ASSERT(result->rows == a->rows);
  TENSOR_ASSERT(result->cols == b->cols);

  int size = a->cols;
  
  for (int i = 0; i < a->rows; ++i) 
  {
    for (int j = 0; j < b->cols; ++j) 
    {
      TENSOR_AT(result, i, j) = 0;
      for (int k = 0; k < size; ++k)
      {
        TENSOR_AT(result, i, j) += TENSOR_AT(a, i, k) * TENSOR_AT(b, k, j);
      }
    }
  }
  return result;
}

void Tensor_map(Tensor2D* a, float (*func)(float))
{
  for (int i = 0; i < a->rows; ++i)  
  {
    for (int j = 0; j < a->cols; ++j)
    {
      TENSOR_AT(a, i, j) = (*func)(TENSOR_AT(a, i, j));
    }
  }
}


void Tensor_Dim(Tensor2D* t) 
{
  printf("%i x %i", t->rows, t->cols);
}

void Tensor_print(Tensor2D* tensor)
{
  for (int i = 0; i < tensor->rows; ++i) 
  {
    for (int j = 0; j < tensor->cols; ++j)
    {
      printf("%1.4f  ", TENSOR_AT(tensor, i, j));
    }
    printf("\n");
  }
}

void Tensor_free(Tensor2D* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

#endif // MATRIX_IMPLEMENTATION
