#include "../../include/core/atlasEngine.h"
#include "../../include/logger.h"
#include <string>

AtlasEngine::AtlasEngine(AtlasConfig engineConfig) {
	config = engineConfig;
	cpu_arena = nullptr;
	gpu_arena = nullptr;
}

AtlasEngine::~AtlasEngine() {
	if (cpu_arena != nullptr) {
		delete cpu_arena;
		cpu_arena = nullptr;
	}
	if (gpu_arena != nullptr) {
		delete gpu_arena;
		gpu_arena = nullptr;
	}
}

void AtlasEngine::bootMem() {
	Logger::log("Booting Memory Manager...", LogLevel::Log_INFO);
	Logger::log("Model Architecture: " + std::to_string(config.total_layers) + " Total Layers.", LogLevel::Log_INFO);
	Logger::log("Hardware Split: " + std::to_string(config.gpu_layers) + " to GPU, " +
		std::to_string(config.total_layers - config.gpu_layers) + " to CPU.", LogLevel::Log_INFO);
	//calculates the total memory needed for GPU layers
	size_t gpuMemNeeded = config.gpu_layers * config.memperlay;
	int cpuLayers = config.total_layers - config.gpu_layers;
	size_t cpuMemNeeded = cpuLayers * config.memperlay;
	//boot CPU hardware
	Logger::log("[DEBUG] Attempting to reserve " + std::to_string(cpuMemNeeded / (1024 * 1024)) + " MB on CPU...", LogLevel::Log_INFO);
	if(cpuMemNeeded > 0){
		cpu_arena = new Arena(cpuMemNeeded);
	}
	Logger::log("[DEBUG] Attempting to reserve " + std::to_string(gpuMemNeeded / (1024 * 1024)) + " MB on GPU...", LogLevel::Log_INFO);
	if(gpuMemNeeded > 0){
		gpu_arena = new GpuArena(gpuMemNeeded);
	}
}