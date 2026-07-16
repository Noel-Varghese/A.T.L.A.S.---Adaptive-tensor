#pragma once 
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>

class GpuArena {
	private:
		uint8_t* base_ptr;
		size_t capacity;
		size_t currentOffset;
	public:
		GpuArena(size_t size);
		~GpuArena();
		void* allocate(size_t request_size, size_t alignment = 256) {
			size_t algOffset = (currentOffset + (alignment - 1)) & ~(alignment - 1);
			if (algOffset + request_size > capacity) {
				throw std::bad_alloc();//no more VRAM available
			}
			void* res_ptr = base_ptr + algOffset;
			currentOffset = algOffset + request_size;
			return res_ptr;
		}
		void reset() {
			currentOffset = 0;
		}
	
};