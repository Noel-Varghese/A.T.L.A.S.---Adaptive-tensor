#include <iostream>
#include "../include/core/atlasEngine.h"
#include "../include/memory/modelLoader.h"
#include "../include/logger.h"
#include "../include/compute/Math.h"

int main() {
	try {
		Logger::log("     A.T.L.A.S. INFERENCE ENGINE       ", LogLevel::Log_INFO);
		AtlasConfig config;
		config.total_layers = 32;
		config.gpu_layers = 24;
		config.memperlay = 150ULL * 1024ULL * 1024ULL;
		Logger::log("INITIALIZING HARDWARE....", LogLevel::Log_INFO);
		AtlasEngine engine(config);
		//boots memory manager
		engine.bootMem();
		//to test the file loader 
		ModelLoader loader("C:/Users/vargh/Desktop/Noel/Projects/ATLAS/Code/models/Qwen3-8B-Q4_K_M.gguf");
		void* data_ptr = loader.loadChunkToArena(engine.getCPUArena(), 1024);
		Logger::log("Requesting layer 0 query weights", LogLevel::Log_INFO);
		Tensor q_weight = loader.loadTensorName("blk.0.attn_q.weight", engine.getCPUArena(), Device::CPU);
		Logger::log("Requesting Token Embedding...", LogLevel::Log_INFO);
		Tensor EmbedMat = loader.loadTensorName("token_embd.weight", engine.getCPUArena(), Device::CPU);
		Logger::log("Embedding Matrix Shape: " + std::to_string(EmbedMat.rows) + " x " + std::to_string(EmbedMat.cols), LogLevel::Log_INFO);
		Logger::log("SUCCESS: Tensor Hydrated!", LogLevel::Log_INFO);
		Logger::log("Matrix Shape: " + std::to_string(q_weight.rows) + " x " + std::to_string(q_weight.cols), LogLevel::Log_INFO);

		size_t expected_bytes = (q_weight.rows * q_weight.cols / 256) * 104;
		Logger::log("Physical RAM Consumed: " + std::to_string(expected_bytes) + " bytes.", LogLevel::Log_INFO);
		Logger::log("Slicing the Token ID", LogLevel::Log_INFO);
		int tokenID = 1532;
		int blocksPerRow = EmbedMat.cols / 256;
		size_t bytesPerRow = blocksPerRow * 104;
		size_t byteOffset = tokenID * bytesPerRow;
		uint8_t* xVectorPtr = static_cast<uint8_t*>(EmbedMat.data) + byteOffset;
		Logger::log("Successfully isolated X vector at byte offset: " + std::to_string(byteOffset), LogLevel::Log_DEBUG);
		std::cout << "[DEBUG] First 5 raw bytes of the word 'Hello': ";
		for (int i = 0; i < 5; i++) {
			std::cout << (int)xVectorPtr[i] << " ";
		}
		std::cout << "\n";
		Logger::log("SYSTEM INITIALIZED SUCCESSFULLY", LogLevel::Log_DEBUG);
	}
	catch (const std::exception& e) {
		Logger::log(std::string("FATAL CRASH: ")+e.what(), LogLevel::Log_ERROR);
	}
	return 0;
}