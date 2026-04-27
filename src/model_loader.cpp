#include <iostream>
#include <string>
#include <windows.h>
#include <fstream>
#include <cstdint>
#include "../include/logger.h"

int main(){
    const char* model_path = "models/Qwen3-8B-Q4_K_M.gguf";
    Logger::log("Booting A.T.L.A.S. ", LogLevel::Log_INFO);
    Logger::log(std::string("attempting to open: ")+model_path, LogLevel::Log_DEBUG);
    //helps use the file without windows interupting
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        Logger::log("Could not open model file. Check the path.", LogLevel::Log_ERROR);
        return 1;
    }

    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    //helps calculate file size
    double size_in_gb = (double)fileSize.QuadPart/(1024*1024*1024);
    Logger::log("Brain Size Detected: " + std::to_string(size_in_gb) + " GB", LogLevel::Log_INFO);
    //creats an internal OS structure needed to tread an SSD as a RAM
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (hMapping == NULL) {
        Logger::log("Kernel failed to create memory mapping.", LogLevel::Log_ERROR);
        CloseHandle(hFile);
        return 1;
    }
    //makes the CPU think its looking at the map but its actually looking at the SSD
    void* mapped_data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (mapped_data == NULL) {
        Logger::log("Failed to map view of file into Virtual RAM.", LogLevel::Log_ERROR);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }

    Logger::log("Zero-Copy Map successful. Virtual Pointer acquired.", LogLevel::Log_INFO);
    char* data_ptr = static_cast<char*>(mapped_data);
    std::string magic(data_ptr, 4); 

    Logger::log("Pointer inspected. First 4 bytes: [" + magic + "]", LogLevel::Log_DEBUG);

    if (magic == "GGUF") {
        Logger::log("A.T.L.A.S. is mapped and ready for tensor alignment.", LogLevel::Log_INFO);
        //Reads data pointer using 32-bit integer
        uint32_t version = *reinterpret_cast<uint32_t*>(data_ptr+4);
        //reads Tensor count
        uint64_t tensor_count = *reinterpret_cast<uint64_t*>(data_ptr + 8);
        //Reads Metadata
        uint64_t metadata_kv = *reinterpret_cast<uint64_t*>(data_ptr+16);
        //Helps print the results
        Logger::log("GGUF Format Version: " + std::to_string(version), LogLevel::Log_DEBUG);
        Logger::log("Neural Network Tensors: " + std::to_string(tensor_count), LogLevel::Log_INFO);
        Logger::log("Hyperparameter KV Pairs: " + std::to_string(metadata_kv), LogLevel::Log_INFO);

    } else {
        Logger::log("Corrupted memory map. Expected GGUF.", LogLevel::Log_ERROR);
    }
    //helps prevent memory leak
    UnmapViewOfFile(mapped_data);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    return 0;
}