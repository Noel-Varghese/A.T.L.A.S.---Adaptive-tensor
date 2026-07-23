#pragma once
#include "arena.h"
#include "../../include/core/tensor.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

struct TensorInfo {
	std::vector<uint64_t> shape;
	uint32_t ggml_type;
	uint64_t absOffset;
};

class ModelLoader {
	private:
		size_t calculateReqBytes(int rows, int cols, DataType dtype);
		std::string filePath;
		std::unordered_map<std::string, TensorInfo> tensor_registry;
		void* mappedData;
	public:
		ModelLoader(const std::string& path);
		//helps read the bytes directly from the SSD into the CPU
		void* loadChunkToArena(Arena* memArena, size_t chunkSize);
		Tensor loadTensor(Arena* targetName, int rows, int cols, DataType dtype, Device loc);
		Tensor loadTensorName(const std::string& tensorName, Arena* targetArena, Device loc);
};
