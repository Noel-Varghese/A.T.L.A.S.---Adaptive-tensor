#include <iostream>
#include <cmath>
#include "../include/core/atlasEngine.h"
#include "../include/memory/modelLoader.h"
#include "../include/logger.h"
#include "../include/compute/Math.h"

float* computeProj(Arena* arena, const Tensor& weight, const float* input){
	float* op = static_cast<float*>(arena->allocate(weight.rows * sizeof(float)));
	int blocksPerRow = weight.cols / 256;
	size_t bytesPerRow = (weight.dtype == DataType::Q6_K) ? blocksPerRow * 210 : blocksPerRow * 144;
	
	for (int row = 0;row < weight.rows;row++) {
		uint8_t* rowPtr = static_cast<uint8_t*>(weight.data) + (row * bytesPerRow);
		op[row] = (weight.dtype == DataType::Q6_K) ? VecDotQ6K_FP32(weight.cols, rowPtr, input) : VecDotQ4KM_FP32(weight.cols, rowPtr, input);
	}
	return op;
}

int main() {
	try {
		Logger::log("-----------------ATLAS ENGINE BOOTING-----------------", LogLevel::Log_INFO);
		//setting up the memory
		AtlasConfig config;
		config.total_layers = 32;
		config.gpu_layers = 24;
		config.memperlay = 150ULL * 1024ULL * 1024ULL; 
		Logger::log("init HARDWARE", LogLevel::Log_INFO);
		AtlasEngine engine(config);
		engine.bootMem();
		//loading model
		ModelLoader loader("C:/Users/vargh/Desktop/Noel/Projects/ATLAS/Code/models/Qwen3-8B-Q4_K_M.gguf");
		//weights of the model
		Tensor qWeight = loader.loadTensorName("blk.0.attn_q.weight", engine.getCPUArena(), Device::CPU);
		Tensor kWeight = loader.loadTensorName("blk.0.attn_k.weight", engine.getCPUArena(), Device::CPU);
		Tensor vWeight = loader.loadTensorName("blk.0.attn_v.weight", engine.getCPUArena(), Device::CPU);
		Tensor oWeight = loader.loadTensorName("blk.0.attn_output.weight", engine.getCPUArena(), Device::CPU);
		//feedForwardNetwork weights
		Tensor ffn_gate = loader.loadTensorName("blk.0.ffn_gate.weight", engine.getCPUArena(), Device::CPU);
		Tensor ffn_up = loader.loadTensorName("blk.0.ffn_up.weight", engine.getCPUArena(), Device::CPU);
		Tensor ffn_down = loader.loadTensorName("blk.0.ffn_down.weight", engine.getCPUArena(), Device::CPU);
		Tensor embedMat = loader.loadTensorName("token_embd.weight", engine.getCPUArena(), Device::CPU);
		//Tokenizer
		const int tokenID = 1532;
		float* Xvec = nullptr;
		if (embedMat.dtype == DataType::FP32) {
			Logger::log("Embedding is FP32. Direct pointer slice...", LogLevel::Log_INFO);
			size_t bytesPerRow = embedMat.cols * sizeof(float);
			size_t byteOffset = tokenID * bytesPerRow;
			Xvec = reinterpret_cast<float*>(static_cast<uint8_t*>(embedMat.data) + byteOffset);
		}
		else if (embedMat.dtype == DataType::Q4_K_M) {
			Logger::log("Embedding is Q4_K_M. Dequantizing...", LogLevel::Log_INFO);
			int blocksPerRow = embedMat.cols / 256;
			size_t bytesPerRow = blocksPerRow * 144;
			size_t byteOffset = tokenID * bytesPerRow;
			uint8_t* compressedPtr = static_cast<uint8_t*>(embedMat.data) + byteOffset;
			Xvec = static_cast<float*>(engine.getCPUArena()->allocate(embedMat.cols * sizeof(float)));
			for (int i = 0;i < blocksPerRow; i++) {
				dequantize(compressedPtr + (i*144), Xvec + (i*256));
			}
		}
		std::cout << "[DEBUGING] X[0...2]: " << Xvec[0] << ", " << Xvec[2] << "\n";
		Logger::log("Computing Q, K, V projections...", LogLevel::Log_INFO);
		float* Qvec = computeProj(engine.getCPUArena(), qWeight, Xvec);
		float* Kvec = computeProj(engine.getCPUArena(), kWeight, Xvec);
		float* Vvec = computeProj(engine.getCPUArena(), vWeight, Xvec);

		const int currentTokenPos = 0;
		const int numQHeads = 32;
		const int headDim = qWeight.cols / numQHeads;
		Logger::log("Applying RoPE to " + std::to_string(numQHeads) +
			" heads (head dim " + std::to_string(headDim) + ")...", LogLevel::Log_INFO);

		for (int h = 0; h < numQHeads; h++) {
			ROPE(Qvec + (h * headDim), currentTokenPos,	headDim);
			ROPE(Kvec + (h * headDim), currentTokenPos,	headDim);
		}

		Logger::log("Executing attention...", LogLevel::Log_INFO);
		float* attnOut = static_cast<float*>(engine.getCPUArena()->allocate(qWeight.cols * sizeof(float)));
		for (int h = 0; h < numQHeads; h++) {
			AttentionSingleToken(
				Qvec + (h * headDim),
				Kvec + (h * headDim),
				Vvec + (h * headDim),
				attnOut + (h * headDim),
				headDim
			);
		}

		std::cout << "[DEBUG] Attention out[0..2]: "
			<< attnOut[0] << ", " << attnOut[1] << ", " << attnOut[2] << "\n";

		float* Ovec = computeProj(engine.getCPUArena(), oWeight, attnOut);

		Logger::log("applying residual connection X=X+0 ....", LogLevel::Log_INFO);
		for (int i = 0; i < embedMat.cols;i++) {
			Xvec[i] += Ovec[i];
		}

		std::cout << "[DEBUG] X after attention residual [0..2]: "
			<< Xvec[0] << ", " << Xvec[1] << ", " << Xvec[2] << "\n";

		//feed Forward Networking (SwiGlu)

		float* gateVec = computeProj(engine.getCPUArena(), ffn_gate, Xvec);
		float* upVec = computeProj(engine.getCPUArena(), ffn_up, Xvec);

		Logger::log("Computing FFN gate/up projections...", LogLevel::Log_INFO);
		int ffnHiddenDim = ffn_gate.rows;
		for (int i = 0; i < ffnHiddenDim; i++) {
			gateVec[i] = gateVec[i] / (1.0f + std::exp(-gateVec[i]));
		}
		for (int i = 0; i < ffnHiddenDim; i++) {
			gateVec[i] *= upVec[i];
		}
		float* downVec = computeProj(engine.getCPUArena(), ffn_down, gateVec);
		
		for (int i = 0;i < embedMat.cols;i++) {
			Xvec[i] += downVec[i];
		}
		Logger::log("SYSTEM INITIALIZED SUCCESSFULLY -- layer 0 complete", LogLevel::Log_DEBUG);
	}
	catch (const std::exception& e) {
		Logger::log("Error occurred: " + std::string(e.what()), LogLevel::Log_ERROR);
	}
}

//nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/arch:AVX2 /EHsc /std:c++17" src/main.cpp src/core/Atlas_Engine.cpp src/memory/modelLoader.cpp src/memory/arena.cpp src/memory/gpuArena.cu src/compute/math.cpp src/compute/math.cu src/Logger.cpp -o atlas.exe