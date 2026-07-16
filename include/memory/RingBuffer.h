#pragma once
#include <cstdint>
#include <cstddef>
#include <iostream>

class KVCacheRingBuffer {
	private:
		float* cacheMem;
		size_t maxToken;
		size_t TokenSizeBytes;
		size_t currentTokenIdx;
	public:
		KVCacheRingBuffer(void* allocatedMem, size_t maxCapacity, size_t tokenSize) {
			cacheMem = static_cast<float*>(allocatedaMem);
			maxToken = maxCapacity;
			TokenSizeBytes = tokenSize;
			currentTokenIdx = 0;
		}
		float* getNextSlot() {
			size_t ringIdx = currentTokenIdx % maxToken;
			size_t offsetEle = (ringIdx * TokenSizeBytes) / sizeof(float);
			currentTokenIdx++;
			return cacheMem + offsetEle;
		}
		size_t getActiveTokens() {
			return (currentTokenIdx < maxToken) ? currentTokenIdx : maxToken;
		}
};
