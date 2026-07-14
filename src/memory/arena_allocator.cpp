#include <iostream>
#include <cstdint>

class ArenaAllocator {
	private:
		size_t capacity;
		size_t current_offset;
		uint8_t* base_ptr;
	public:
		//to ask the operating system for a big block of memory at once
		ArenaAllocator(size_t sizeInBytes){
			capacity = sizeInBytes;
			current_offset = 0;
			base_ptr = new uint8_t[capacity];//to allocate tht memory
			std::cout << "Arena Boosted: " << capacity / (1024 * 1024) << " MB recieved\n";
		}
		~ArenaAllocator() {
			delete[] base_ptr;
		}
		void* allocate(size_t requested_bytes) {
			if (current_offset + requested_bytes > capacity) {
				std::cout << "ArenaAllocator: Out of memory!\n";
				return nullptr;
			}
			void* memory_add = base_ptr + current_offset;
			current_offset += requested_bytes;
			return memory_add;
		}
		void reset() {
			current_offset = 0;
		}
};