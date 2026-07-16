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
		Logger::log("SYSTEM INITIALIZED SUCCESSFULLY", LogLevel::Log_DEBUG);
	}
	catch (const std::exception& e) {
		Logger::log(std::string("FATAL CRASH: ")+e.what(), LogLevel::Log_ERROR);
	}
	return 0;
}