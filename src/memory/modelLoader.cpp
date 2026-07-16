#include "../../include/memory/modelLoader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

ModelLoader::ModelLoader(const std::string& path) {
	filePath = path;
}

void* ModelLoader::loadChunkToArena(Arena* memArena, size_t chunkSize) {
	//to open the file in raw binary
	std::ifstream file(filePath, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + filePath);
	}
	void* dest_ptr = memArena->allocate(chunkSize);
	std::cout << "[Model Loader] Reading " << chunkSize / (1024 * 1024) << " MB from SSD directly to Arena...\n";
	//helps move the bytes from SSD to RAM
	file.read(reinterpret_cast<char*>(dest_ptr), chunkSize);
	if (!file) {
		throw std::runtime_error("Failed to read the requested chunk from file: ");
	}
	file.close();
	std::cout << "[Model Loader] Successfully loaded binary block into Arena.\n";
	return dest_ptr;
}