#pragma once
#include "../memory/arena.h"
#include "../memory/gpu_arena.h"

//config profile
struct AtlasConfig {
	int total_layers;
	int gpu_layers;
	size_t memperlay;
};

class AtlasEngine {
	private:
		Arena* cpu_arena;
		GpuArena* gpu_arena;
		AtlasConfig config;
	public:
		AtlasEngine(AtlasConfig engineConfig);
		~AtlasEngine();
		void bootMem();
		Arena* getCPUArena() { return cpu_arena; }

};
