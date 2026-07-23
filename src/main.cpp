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
		Tensor q_weight = loader.loadTensorName("model.layers.0.self_attention.query.weight", engine.getCPUArena(), Device::CPU);
		Logger::log("SUCCESS: Tensor Hydrated!", LogLevel::Log_INFO);
		Logger::log("Matrix Shape: " + std::to_string(q_weight.rows) + " x " + std::to_string(q_weight.cols), LogLevel::Log_INFO);

		size_t expected_bytes = (q_weight.rows * q_weight.cols / 256) * 104;
		Logger::log("Physical RAM Consumed: " + std::to_string(expected_bytes) + " bytes.", LogLevel::Log_INFO);
		Logger::log("SYSTEM INITIALIZED SUCCESSFULLY", LogLevel::Log_DEBUG);
	}
	catch (const std::exception& e) {
		Logger::log(std::string("FATAL CRASH: ")+e.what(), LogLevel::Log_ERROR);
	}
	return 0;
}