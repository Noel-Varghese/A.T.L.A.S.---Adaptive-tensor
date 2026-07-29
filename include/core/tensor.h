#pragma once 
#include <cstdint>

enum class Device {
	CPU,
	GPU
};

enum class DataType {
	FP32,
	Q4_K_M,
	Q6_K
};

struct Tensor {
	int rows, cols;
	int stride;
	DataType dtype;
	Device location;
	void* data;
};
