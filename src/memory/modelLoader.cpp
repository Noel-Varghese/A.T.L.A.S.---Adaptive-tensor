#include "../../include/memory/modelLoader.h"
#include "../../include/logger.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <Windows.h>//used for handling memories for the zero-copy mechanism

ModelLoader::ModelLoader(const std::string& path) {
	filePath = path;
	Logger::log("Mapping GGUF into OS Virtual Memory...", LogLevel::Log_INFO);

	HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open model file.");

	HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hMapping == NULL) throw std::runtime_error("Failed to create Memory map");

	mappedData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
	if (mappedData == NULL) throw std::runtime_error("Failed to map view of file.");

	char* dataPtr = static_cast<char*>(mappedData);
	if (std::string(dataPtr, 4) != "GGUF") {
		throw std::runtime_error("Invalid GGUF file format.");
	}

	uint64_t tensorCount = *reinterpret_cast<uint64_t*>(dataPtr + 8);
	uint64_t metadataKV = *reinterpret_cast<uint64_t*>(dataPtr + 16);
	uint64_t offset = 24;

	Logger::log("Fast-forwarding through " + std::to_string(metadataKV) + " KV pairs...", LogLevel::Log_DEBUG);


	for (uint64_t i = 0; i < metadataKV; ++i) {
		uint64_t keyLen = *reinterpret_cast<uint64_t*>(dataPtr + offset); // FIXED CAST
		offset += 8 + keyLen;

		uint32_t valType = *reinterpret_cast<uint32_t*>(dataPtr + offset);
		offset += 4;

		if (valType == 4 || valType == 5 || valType == 6) {
			offset += 4;
		}
		else if (valType == 7) {
			offset += 1;
		}
		else if (valType == 8) {
			uint64_t str_len = *reinterpret_cast<uint64_t*>(dataPtr + offset);
			offset += 8 + str_len;
		}
		else if (valType == 9) {
			uint32_t arr_type = *reinterpret_cast<uint32_t*>(dataPtr + offset); // FIXED dataPtr
			uint64_t arr_len = *reinterpret_cast<uint64_t*>(dataPtr + offset + 4);
			offset += 12;
			for (uint64_t a = 0; a < arr_len; ++a) {
				if (arr_type == 4 || arr_type == 5 || arr_type == 6) offset += 4;
				else if (arr_type == 7) offset += 1;
				else if (arr_type == 8) {
					uint64_t slen = *reinterpret_cast<uint64_t*>(dataPtr + offset);
					offset += 8 + slen;
				}
			}
		}
		else {
			offset += 4;
		}
}

		Logger::log("Mapping " + std::to_string(tensorCount) + " tensors to registry...", LogLevel::Log_INFO);

		for (uint64_t i = 0; i < tensorCount; ++i) {
			uint64_t name_len = *reinterpret_cast<uint64_t*>(dataPtr + offset);
			offset += 8;
			std::string t_name(dataPtr + offset, name_len);
			offset += name_len;

			uint32_t n_dims = *reinterpret_cast<uint32_t*>(dataPtr + offset);
			offset += 4;

			TensorInfo info;
			for (uint32_t d = 0; d < n_dims; ++d) {
				info.shape.push_back(*reinterpret_cast<uint64_t*>(dataPtr + offset));
				offset += 8;
			}

			info.ggml_type = *reinterpret_cast<uint32_t*>(dataPtr + offset);
			offset += 4;
			info.absOffset = *reinterpret_cast<uint64_t*>(dataPtr + offset);
			offset += 8;

			tensor_registry[t_name] = info;
		}

		// Inorder to align the data to 32 bytes
		uint64_t data_section_start = (offset + 31) & ~31ULL;
		for (auto& pair : tensor_registry) {
			pair.second.absOffset += data_section_start;
		}

		Logger::log("Brain fully mapped. Registry size: " + std::to_string(tensor_registry.size()), LogLevel::Log_INFO);
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
	else if (info.ggml_type == 12) {
		mapped_dtype = DataType::Q4_K_M;
	}
	else if (info.ggml_type == 14) {
		mapped_dtype = DataType::Q6_K;
	}
	else {
		throw std::runtime_error("Unsupported ggml_type: " + std::to_string(info.ggml_type));
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
		return numBlocks * 144;
	}else if(dtype == DataType::Q6_K){
				if (totalElements % 256 != 0) {
			throw std::invalid_argument("For Q6_K, total elements must be a multiple of 256.");
		}
		size_t numBlocks = totalElements / 256;
		return numBlocks * 210;
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
