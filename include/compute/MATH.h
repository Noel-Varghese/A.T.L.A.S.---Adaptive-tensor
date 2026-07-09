#ifndef ATLAS_MATH_H
#define ATLAS_MATH_H
float avx2_dotProduct(const float* vec_A, const float* vec_B, int size);
void matvec_fp32(const float* matrix_W, const float* vec_X, float* vec_Y, int rows, int cols);
#endif