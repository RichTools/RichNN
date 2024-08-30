#ifndef _VECTOR_H_
#define _VECTOR_H_

typedef struct {
  int dimension;
  float* data;
} Vector;

Vector* Vector_alloc(int dim);
void Vector_free(Vector* vec);
Vector* Vector_init(int dim);

Vector* Vector_dot(Vector* a, Vector* b);

#endif // _VECTOR_H_

#ifndef VECTOR_ASSERT
#include <assert.h>
define VECTOR_ASSERT assert
#endif

#ifdef VECTOR_IMPLEMETATION

Vector* Vector_alloc(int dim)
{
  Vector* vector = (Vector*)malloc(sizeof(Vector));

  if (!vector)
  {
    fprintf(stderr, "Vector Allocation Failed");
    exit(1);
  }

  vector->dimension = dim;
  vector->data = (float*)malloc(dim * sizeof(float));

  if (!vector->data) 
  {
    fprintf(stderr, "Failed to Allocate Memory for Data\n");
    free(vector);
    exit(1);
  }
}

Vector* Vector_init(int dim)
{
  Vector* vector = Vector_alloc(dim);

  for (int i = 0; i < dim; ++i)
  {
    vector->data[i] = 0;
  }
}

Vector* Vector_dot(Vector* a, Vector* b) 
{
  float prod = 0.0f;
  VECTOR_ASSERT(a->dim == b->dim);
  for (int i = 0; i < a->dim; ++i) 
  {
    for (int j = 0; j < b->dim; ++j)
    {
      prod += (a->data[i] * b->data[j]);
    }
  }
  return prod;
}

void Vector_free(Vector* vector)
{
  free(vector->data);
  free(vector);
}

#endif // VECTOR_IMPLEMETATION
