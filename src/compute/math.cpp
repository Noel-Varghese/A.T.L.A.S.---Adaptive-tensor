#include <immintrin.h>//used for the AVX calculation
#include "../../include/compute/MATH.h"
#include <cstring>
#include <cmath>

float FP16ToFP32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t expo = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) {
            f = sign;
        }
        else {
            expo = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                expo--;
            }
            mant &= 0x03FF;
            f = sign | (expo << 23) | (mant << 13);
        }
    }
    else if (expo == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    }
    else {
        f = sign | ((expo - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

static inline void GetScaleK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize(const uint8_t* compressedBlock, float* OutFloatz) {
    uint16_t d_Raw, d_Min;
    std::memcpy(&d_Raw, compressedBlock + 0, sizeof(uint16_t));
    std::memcpy(&d_Min, compressedBlock + 2, sizeof(uint16_t));

    const float d = FP16ToFP32(d_Raw);
    const float dMin = FP16ToFP32(d_Min);
    const uint8_t* scales = compressedBlock + 4;
    const uint8_t* q = compressedBlock + 16;

    int i = 0;
    float* y = OutFloatz;

    //For 256 elements = 4 iterations of 64 elements each
    for (int c = 0; c < 256; c += 64) {
        uint8_t sc, m;
        GetScaleK4(i + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dMin * m;
        GetScaleK4(i + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dMin * m;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * (q[l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * (q[l] >> 4) - m2;
        }

        q += 32;
        i += 2;

    }
}

float avx2_dotProduct(const float* vec_A, const float* vec_B, int size){
    float sum = 0.0f;
    int i = 0;
    //to create a register filled with zeros to act as an accumulator
    __m256 acc = _mm256_setzero_ps();
    //inorder to go through the 8 floats at a time
    for(; i<= size - 8;i+=8){
        __m256 a = _mm256_load_ps(&vec_A[i]);
        __m256 b = _mm256_load_ps(&vec_B[i]);
        //Multiply-add on the float numbers acc = (a*b)+acc
        acc = _mm256_fmadd_ps(a, b, acc);
    }
    //sice we have 8 partial sums in a 256-bit register
    //we need to extract them back to normal memory to add them up
    alignas(32) float temp[8];
    _mm256_store_ps(temp, acc);
    for(int j=0;j<8;++j){
        sum += temp[j];
    } 
    //incase the matrix isnt divisible by 8
    for(; i<size;++i){
        sum+= vec_A[i]*vec_B[i];
    }
    return sum;
}

float VecDotQ4KM_FP32(const int cols, const uint8_t* wRowCompressed, const float* XVec) {
    float TotalDot = 0.0f;
    int numBlocks = cols / 256;
    alignas(32) float tempFloatz[256];
    for (int b = 0; b < numBlocks;b++) {
        const uint8_t* BlockPtr = wRowCompressed + (b * 144);
        uint16_t D_RAW, D_MIN;
        std::memcpy(&D_RAW, BlockPtr + 0, sizeof(uint16_t));
        std::memcpy(&D_MIN, BlockPtr + 2, sizeof(uint16_t));
        const float d = FP16ToFP32(D_RAW);
        const float dMIN = FP16ToFP32(D_MIN);
        const uint8_t* scales = BlockPtr + 4;
        const uint8_t* q = BlockPtr + 16;
        int i = 0;
        float* y = tempFloatz;
        for (int c = 0;c < 256;c += 64) {
            uint8_t sc, m;
            if (i < 4) {
                sc = scales[i] & 63;
                m = scales[i + 4] & 63;
            }
            else {
                sc = (scales[i + 4] & 0x0F) | ((scales[i - 4] >> 6) << 4);
                m = (scales[i + 4] >> 4) | ((scales[i] >> 6) << 4);
            }
            const float d1 = d * sc;const float m1 = dMIN * m;
            if (i + 1 < 4) { sc = scales[i + 1] & 63; m = scales[i + 5] & 63; }
            else {
                sc = (scales[i + 5] & 0x0F) | ((scales[i - 3] >> 6) << 4);
                m = (scales[i + 5] >> 4) | ((scales[i + 1] >> 6) << 4);
            }
            const float d2 = d * sc;
            const float m2 = dMIN * m;
            for (int l = 0; l < 32; ++l) { *y++ = d1 * (q[l] & 0x0F) - m1; }
            for (int l = 0; l < 32; ++l) { *y++ = d2 * (q[l] >> 4) - m2; }
            q += 32; i += 2;
        }
        float BlockDot = avx2_dotProduct(tempFloatz, XVec + (b * 256), 256);
        TotalDot += BlockDot;
    }
    return TotalDot;
}

void matvec_fp32(const float* matrix_W, const float* vec_X, float* vec_Y, int rows, int cols){
    for(int i=0;i<rows;++i){
        const float* current_row_pointer =  &matrix_W[i*cols];
        vec_Y[i] = avx2_dotProduct(current_row_pointer, vec_X, cols);
    }
}

void dequantizeQ6K(const uint8_t* compressedBlock, float* OutFloatz) {
	const uint8_t* ql = compressedBlock;
	const uint8_t* qh = compressedBlock + 128;
	const int8_t* sc = reinterpret_cast<const int8_t*>(compressedBlock + 192);
    uint16_t DRAW;
    std::memcpy(&DRAW, compressedBlock + 208, sizeof(uint16_t));
	const float d = FP16ToFP32(DRAW);
    float* y = OutFloatz;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;

            int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

float VecDotQ6K_FP32(const int cols, const uint8_t* wRowCompressed, const float* XVec) {
	float TotalDot = 0.0f;
	int numBlocks = cols / 256;
	alignas(32) float tempFloatz[256];
	for (int b = 0; b < numBlocks; b++) {
		const uint8_t* BlockPtr = wRowCompressed + (b * 210);
		dequantizeQ6K(BlockPtr, tempFloatz);
		float BlockDot = avx2_dotProduct(tempFloatz, XVec + (b * 256), 256);
		TotalDot += BlockDot;
	}
	return TotalDot;
}

void ROPE(float* vec, int pos, int headDim, float base) {
    int halfDim = headDim / 2;
    for (int i = 0;i < halfDim;++i) {
        float freq = 1.0f / std::pow(base, (2.0f * i)/headDim);
        float theta = pos * freq;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);
        float x0 = vec[i];
        float x1 = vec[i + halfDim];
        vec[i] = x0 * cosTheta - x1 * sinTheta;
        vec[i + halfDim] = x0 * sinTheta + x1 * cosTheta;
    }
}

void AttentionSingleToken(float* q_head, float* k_head, float* v_head, float* out_head, int head_dim) {
    float score = avx2_dotProduct(q_head, k_head, head_dim);
    score /= std::sqrt(static_cast<float>(head_dim));
    float prob = 1.0f;
    for (int i = 0; i < head_dim;i++) {
        out_head[i] = prob * v_head[i];
    }
}
