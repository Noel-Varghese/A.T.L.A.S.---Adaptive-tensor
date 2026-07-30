#ifndef ATLAS_MATH_H
#define ATLAS_MATH_H
#include <cstdint>
float avx2_dotProduct(const float* vec_A, const float* vec_B, int size);
void matvec_fp32(const float* matrix_W, const float* vec_X, float* vec_Y, int rows, int cols);
//unzips compressed Q4_K_M blocks standard 32-bit floats
void dequantize(const uint8_t* compressedBlock, float* OutFloatz);
float VecDotQ4KM_FP32(const int cols, const uint8_t* wRowCompressed, const float* XVec);
void dequantizeQ6K(const uint8_t* compressedBlock, float* OutFloatz);
float VecDotQ6K_FP32(const int cols, const uint8_t* wRowCompressed, const float* XVec);
void ROPE(float* vec, int pos, int headDim, float base = 10000.0f);
void AttentionSingleToken(float* q_head, float* k_head, float* v_head, float* out_head, int head_dim);
#endif