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
		//Loading the model 
		ModelLoader loader("C:/Users/vargh/Desktop/Noel/Projects/ATLAS/Code/models/Qwen3-8B-Q4_K_M.gguf");
		void* data_ptr = loader.loadChunkToArena(engine.getCPUArena(), 1024);
		//Loading Layer 0
		Tensor q_weight = loader.loadTensorName("blk.0.attn_q.weight", engine.getCPUArena(), Device::CPU);
		Tensor EmbedMat = loader.loadTensorName("token_embd.weight", engine.getCPUArena(), Device::CPU);
		size_t expected_bytes = (q_weight.rows * q_weight.cols / 256) * 144;
		//dequantize token embedding
		int tokenID = 1532;
		float* XVectorFloat = nullptr;
		if (EmbedMat.dtype == DataType::FP32) {
			Logger::log("Embedding is FP32. No decompression needed. Direct pointer slice...", LogLevel::Log_INFO);
			size_t bytesPerRow = EmbedMat.cols *sizeof(float);
			size_t byteOffset = tokenID * bytesPerRow;
			XVectorFloat = reinterpret_cast<float*>(static_cast<uint8_t*>(EmbedMat.data)+byteOffset);
		}
		else if(EmbedMat.dtype == DataType::Q4_K_M){
			Logger::log("Embedding is Q4_K_M. Unzipping 4-bit blocks...", LogLevel::Log_INFO);
			int blocksPerRow = EmbedMat.cols / 256;
			size_t bytesPerRow = blocksPerRow * 144;
			size_t byteOffset = tokenID * bytesPerRow;
			uint8_t* XCompressedPtr = static_cast<uint8_t*>(EmbedMat.data) + byteOffset;
			XVectorFloat = static_cast<float*>(engine.getCPUArena()->allocate(EmbedMat.cols * sizeof(float)));
			Logger::log("Unzipping 4-bit blocks into 32-bit floats via Compute Kernel...", LogLevel::Log_INFO);
			for (int i = 0;i < blocksPerRow;i++) {
				dequantize(XCompressedPtr + (i * 144), XVectorFloat + (i * 256));
			}
		}
		//std::cout << "[DEBUG] First 3 mathematical weights for 'Hello': \n";
		//for (int i = 0; i < 3; i++) {
		//	std::cout << XVectorFloat[i] << "\n";
		//}
		float* YVectorFloat = static_cast<float*>(engine.getCPUArena()->allocate(q_weight.rows * sizeof(float)));
		int WBlockPerRow = q_weight.cols / 256;
		size_t WBytesPerRow = WBlockPerRow * 144;
		for (int r = 0;r < q_weight.rows;r++) {
			uint8_t* WRowPtr = static_cast<uint8_t*>(q_weight.data) + (r * WBytesPerRow);
			YVectorFloat[r] = VecDotQ4KM_FP32(q_weight.cols, WRowPtr, XVectorFloat);
		}
		//Logger::log("SUCCESS: Layer 1 Forward Pass Complete.", LogLevel::Log_INFO);

		//std::cout << "[DEBUG] First 3 mathematical outputs of Layer 1 (Vector Y): \n";
		//for (int i = 0; i < 3; i++) {
		//	std::cout << YVectorFloat[i] << "\n";
		//}

		//Logger::log("Hydrating Layer 1 Key (K) and Value (V) Matrices...", LogLevel::Log_INFO);
		Tensor kWeights = loader.loadTensorName("blk.0.attn_k.weight", engine.getCPUArena(), Device::CPU);
		Tensor vWeights = loader.loadTensorName("blk.0.attn_v.weight", engine.getCPUArena(), Device::CPU);
		float* KVectorFLOAT = static_cast<float*>(engine.getCPUArena()->allocate(kWeights.rows * sizeof(float)));
		float* VVectorFLOAT = static_cast<float*>(engine.getCPUArena()->allocate(vWeights.rows * sizeof(float)));
		int WBlockPerRowK = kWeights.cols / 256;
		int WBytes_PerRow = WBlockPerRowK * 144;
		Logger::log("Initiating AVX2 Math for K Vector (K = X * W_k)...", LogLevel::Log_INFO);
		for (int row = 0;row < kWeights.rows;row++) {
			uint8_t* WRowPtrK = static_cast<uint8_t*>(kWeights.data) + (row * WBytes_PerRow);
			KVectorFLOAT[row] = VecDotQ4KM_FP32(kWeights.cols, WRowPtrK, XVectorFloat);
		}
		Logger::log("Initiating AVX2 Math for V Vector (V = X * W_v)...", LogLevel::Log_INFO);
		int WBlockPerRowV = vWeights.cols / 256;
		size_t WBytes_PerRowV = (vWeights.dtype == DataType::Q6_K) ? WBlockPerRowV * 210 : WBlockPerRowV * 144;

		for (int row = 0; row < vWeights.rows; row++) {
			uint8_t* WRowPtrV = static_cast<uint8_t*>(vWeights.data) + (row * WBytes_PerRowV);
			if (vWeights.dtype == DataType::Q6_K) {
				VVectorFLOAT[row] = VecDotQ6K_FP32(vWeights.cols, WRowPtrV, XVectorFloat);
			}
			else {
				VVectorFLOAT[row] = VecDotQ4KM_FP32(vWeights.cols, WRowPtrV, XVectorFloat);
			}
		}
		//Logger::log("SUCCESS: Q, K, and V vectors are fully calculated.", LogLevel::Log_INFO);
		//std::cout << "[DEBUG] First 3 outputs of Vector K: "
		//	<< KVectorFLOAT[0] << ", " << KVectorFLOAT[1] << ", " << KVectorFLOAT[2] << "\n";
		//std::cout << "[DEBUG] First 3 outputs of Vector V: "
		//	<< VVectorFLOAT[0] << ", " << VVectorFLOAT[1] << ", " << VVectorFLOAT[2] << "\n";
		int currentTokenPos = 0;
		int numQHeads = 32;
		int headDim = q_weight.cols / numQHeads;//4096/32 = 128
		Logger::log("Applying RoPE to 32 Attention Heads (Head Dim: " + std::to_string(headDim) + ")...", LogLevel::Log_INFO);
		for (int h = 0; h < numQHeads; h++) {
			float* QHeadPtr = YVectorFloat + (h * headDim);
			float* KHeadPtr = KVectorFLOAT + (h * headDim);
			ROPE(QHeadPtr, currentTokenPos, headDim);
			ROPE(KHeadPtr, currentTokenPos, headDim);
		}
		Logger::log("SUCCESS: Multi-Head Time vectors successfully rotated.", LogLevel::Log_INFO);
		Logger::log("Executing Scaled Dot-Product Attention...", LogLevel::Log_INFO);
		float* AttentionOUTVector = static_cast<float*>(engine.getCPUArena()->allocate(4096 * sizeof(float)));
		for (int i = 0;i < numQHeads;i++) {
			float* q_head_ptr = YVectorFloat + (i * headDim);
			float* k_head_ptr = KVectorFLOAT + (i * headDim);
			float* v_head_ptr = VVectorFLOAT + (i * headDim);
			float* out_head_ptr = AttentionOUTVector + (i * headDim);
			AttentionSingleToken(q_head_ptr, k_head_ptr, v_head_ptr, out_head_ptr, headDim);
		}
		std::cout << "[DEBUG] First 3 values of the Attention Output Vector: \n"<< AttentionOUTVector[0] << "\n"<< AttentionOUTVector[1] << "\n"<< AttentionOUTVector[2] << "\n";
		Logger::log("Hydrating Attention Output Projection Matrix (O)...", LogLevel::Log_INFO);
		Tensor OWeight = loader.loadTensorName("blk.0.attn_output.weight", engine.getCPUArena(), Device::CPU);
		float* ProjectAttention = static_cast<float*>(engine.getCPUArena()->allocate(OWeight.rows * sizeof(float)));
		Logger::log("Initiating AVX2 Math for Output Projection (O = AttentionOut * W_o)...", LogLevel::Log_INFO);
		if (OWeight.dtype == DataType::Q4_K_M) {
			int OBlocksPerRow = OWeight.cols / 256;
			size_t OBytesPerRow = OBlocksPerRow * 144;
			for (int row = 0;row < OWeight.rows;row++) {
				uint8_t* WRowPtr = static_cast<uint8_t*>(OWeight.data) + (row * OBytesPerRow);
				ProjectAttention[row] = VecDotQ4KM_FP32(OWeight.cols, WRowPtr, AttentionOUTVector);
			}
		}
		else {
			Logger::log("FATAL: Unhandled O matrix format. Check hardware ID.", LogLevel::Log_ERROR);
			return -1;
		}
		Logger::log("Applying Residual Connection (X = X + O)...", LogLevel::Log_INFO);
		std::cout << "[DEBUG] First 3 values of the newly updated Vector X: \n"
			<< XVectorFloat[0] << "\n"
			<< XVectorFloat[1] << "\n"
			<< XVectorFloat[2] << "\n";
		Logger::log("SYSTEM INITIALIZED SUCCESSFULLY", LogLevel::Log_DEBUG);
		//Logger::log("V weight dtype: " + std::to_string(static_cast<int>(vWeights.dtype)), LogLevel::Log_INFO);
	}
	catch (const std::exception& e) {
		Logger::log(std::string("FATAL CRASH: ")+e.what(), LogLevel::Log_ERROR);
	}
	return 0;
}




//nvcc - O3 - arch = sm_89 - std = c++17 - Xcompiler "/arch:AVX2 /EHsc /std:c++17" src / main.cpp src / core / Atlas_Engine.cpp src / memory / modelLoader.cpp src / memory / arena.cpp src / memory / gpuArena.cu src / compute / math.cpp src / compute / math.cu src / Logger.cpp - o atlas.exe