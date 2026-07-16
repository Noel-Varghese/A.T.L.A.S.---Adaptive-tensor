#include "../../include/memory/gpu_arena.h"
#include <iostream>

GpuArena::GpuArena(size_t size){
	capacity = size;
	currentOffset = 0;
	//attempt to allocate memory on the GPU VRAM
	cudaError_t err = cudaMalloc((void**)&base_ptr, capacity);
	if(err != cudaSuccess){
		std::cout << "FATAL: CUDA Malloc Failed! You requested more VRAM than the GPU has.\n";
        std::cout << "NVIDIA Error: " << cudaGetErrorString(err) << "\n";
        throw std::bad_alloc();
	}
	std::cout << "[GPU Arena] Booted: " << capacity / (1024 * 1024) << " MB VRAM reserved.\n";
	
}
GpuArena::~GpuArena(){
		cudaFree(base_ptr);
}