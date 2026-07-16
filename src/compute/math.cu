#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_multiply_kernel(const float* vec_A, const float* vec_B, float* vec_Y, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<size){
        vec_Y[index] = vec_A[index]*vec_B[index];
    }
    
}
// int main(){
//     int size = 4096;
//     size_t byte_size = size * sizeof(float);
//     //inorder to allocate CPU RAM
//     float* host_a = new float[size];
//     float* host_b = new float[size];
//     float* host_y = new float[size]; 
//     for(int i=0;i<size;i++){
//         host_a[i] = 2.0f;
//         host_b[i] = 3.0f;
//     }
//     //inorder to allocate CPU VRAM
//     float *deviceA, *deviceB, *deviceY;
//     cudaMalloc(&deviceA, byte_size);
//     cudaMalloc(&deviceB, byte_size);
//     cudaMalloc(&deviceY, byte_size);

//     //copying CPU->GPU
//     cudaMemcpy(deviceA, host_a, byte_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(deviceB, host_b, byte_size, cudaMemcpyHostToDevice);
//     //launch kernel
    
//     int threads = 256;
//     int blocks = (size + threads - 1)/threads;
//     vector_multiply_kernel<<<blocks, threads>>>(deviceA, deviceB, deviceY, size);

//     cudaDeviceSynchronize();
//     //GPU->CPU
//     cudaMemcpy(host_y, deviceY, byte_size, cudaMemcpyDeviceToHost);
//     std::cout << "--- A.T.L.A.S. CUDA CORE AWAKE ---\n";
//     std::cout << "GPU Math Check: Index 0 is " << host_y[0] << " (Expected 6)\n";
//     std::cout << "GPU Math Check: Index 4095 is " << host_y[4095] << " (Expected 6)\n"; 
//     cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceY);
//     delete[] host_a; delete[] host_b; delete[] host_y;

//     return 0;
// }