#include <immintrin.h>//used for the AVX calculation

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