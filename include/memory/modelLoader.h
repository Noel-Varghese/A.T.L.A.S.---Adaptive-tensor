#pragma once
#include "arena.h"
#include <string>
#include <cstddef>

class ModelLoader {
	private:
		std::string filePath;
	public:
		ModelLoader(const std::string& path);
		//helps read the bytes directly from the SSD into the CPU
		void* loadChunkToArena(Arena* memArena, size_t chunkSize);
};
