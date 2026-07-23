#include "../../include/memory/modelLoader.h"
#include "../../include/logger.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <Windows.h>

ModelLoader::ModelLoader(const std::string& path) {
	filePath = path;
	Logger::log("Mapping GGUF into OS Virtual Memory...", LogLevel::Log_INFO);
	HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open model file.");
	HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hMapping == NULL)throw std::runtime_error("Failed to create Memory map");
	mappedData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
	if (mappedData == NULL)throw std::runtime_error("Failed to map view of file.");

}

Tensor ModelLoader::loadTensorName(const std::string& name, Arena* targetArena, Device loc) {
	auto it = tensor_registry.find(name);
	if (it == tensor_registry.end()) {
		Logger::log("FATAL: Architecture mismatch. Tensor not found -> " + name, LogLevel::Log_ERROR);
		throw std::runtime_error("Tensor missing from GGUF registry: " + name);
	}
	const TensorInfo& info = it->second;
	int cols = info.shape.size() > 0 ? info.shape[0] : 1;
	int rows = info.shape.size() > 1 ? info.shape[1] : 1;

	DataType mapped_dtype;
	if (info.ggml_type == 0) {
		mapped_dtype = DataType::FP32;
	}
	else {
		mapped_dtype = DataType::Q4_K_M;
	}
	size_t byte_size = calculateReqBytes(rows, cols, mapped_dtype);
	void* dest_ptr = targetArena->allocate(byte_size);
	uint8_t* src_ptr = static_cast<uint8_t*>(mappedData) + info.absOffset;
	std::memcpy(dest_ptr, src_ptr, byte_size);
	Logger::log("Hydrated Tensor: " + name + " into " +
		(loc == Device::CPU ? "CPU RAM" : "GPU VRAM"), LogLevel::Log_DEBUG);
	Tensor t;
	t.rows = rows;
	t.cols = cols;
	t.stride = cols;
	t.dtype = mapped_dtype;
	t.location = loc;
	t.data = dest_ptr;

	return t;
}

void* ModelLoader::loadChunkToArena(Arena* memArena, size_t chunkSize) {
	//to open the file in raw binary
	//std::ifstream file(filePath, std::ios::binary);
	//if (!file.is_open()) {
	//	throw std::runtime_error("Failed to open file: " + filePath);
	//}
	void* dest_ptr = memArena->allocate(chunkSize);
	std::cout << "[Model Loader] Reading " << chunkSize / (1024 * 1024) << " MB from SSD directly to Arena...\n";
	//helps move the bytes from SSD to RAM
	//file.read(reinterpret_cast<char*>(dest_ptr), chunkSize);
	//if (!file) {
	//	throw std::runtime_error("Failed to read the requested chunk from file: ");
	//}
	//file.close();
	//std::cout << "[Model Loader] Successfully loaded binary block into Arena.\n";
	return dest_ptr;
}

size_t ModelLoader::calculateReqBytes(int rows, int cols, DataType dtype) {
	size_t totalElements = static_cast<size_t>(rows) * cols;
	if (dtype == DataType::FP32) {
		return totalElements * sizeof(float);// 4bytes/float
	}
	else if (dtype == DataType::Q4_K_M) {
		if (totalElements % 256 != 0) {
			throw std::invalid_argument("For Q4_K_M, total elements must be a multiple of 256.");
		}
		size_t numBlocks = totalElements / 256;
		return numBlocks * 104;
	}
	throw std::invalid_argument("Unsupported data type for tensor loading.");

}

Tensor ModelLoader::loadTensor(Arena* targetArena, int rows, int cols, DataType dtype, Device loc) {
	size_t reqBytes = calculateReqBytes(rows, cols, dtype);
	void* dataPtr = targetArena->allocate(reqBytes);
	Tensor t;
	t.rows = rows;
	t.cols = cols;
	t.stride = cols; // Assuming stride is equal to cols for simplicity
	t.dtype = dtype;
	t.location = loc;
	t.data = dataPtr;
	return t;
}
