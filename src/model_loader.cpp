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
        
        //help start the scanner right after the static header
        uint64_t offset = 24;
        //Reads the length of first key
        Logger::log("Initiating Metadata Scanner", LogLevel::Log_INFO);
        for(uint64_t i = 0;i<metadata_kv ;++i){
            //reads the key length
            uint64_t key_length = *reinterpret_cast<uint64_t*>(data_ptr + offset);
            offset += 8;
            //Extracts Key Name
            std::string key_name(data_ptr + offset, key_length);
            offset += key_length;

            //Read value type ID
            uint32_t value_type = *reinterpret_cast<uint32_t*>(data_ptr + offset);
            offset += 4;
            //Extracts value based on type
            std::string print_val = "";
            if(value_type == 4 || value_type == 5){
                //32 bit integers
                uint32_t val = *reinterpret_cast<uint32_t*>(data_ptr + offset);
                print_val = std::to_string(val);
                offset += 4;
            }else if(value_type == 6){//32 bit float
                float val = *reinterpret_cast<float*>(data_ptr + offset);
                print_val = std::to_string(val);
                offset += 4;
            }else if (value_type == 7) { // Boolean
                bool val = *reinterpret_cast<bool*>(data_ptr + offset);
                print_val = val ? "true" : "false";
                offset += 1;
            }
            else if (value_type == 8) { // String
                uint64_t str_len = *reinterpret_cast<uint64_t*>(data_ptr + offset);
                offset += 8;
                print_val = std::string(data_ptr + offset, str_len);
                offset += str_len;
            }
            else if (value_type == 9) { // Array
                print_val = "[ARRAY DETECTED - HALTING SCAN TO PREVENT OVERFLOW]";
                Logger::log(key_name + " = " + print_val, LogLevel::Log_DEBUG);
                break; // We break the loop here until we write the array logic
            }
            else {
                print_val = "[UNKNOWN TYPE ID: " + std::to_string(value_type) + "]";
                Logger::log(key_name + " = " + print_val, LogLevel::Log_ERROR);
                break; // Safety halt
            }   
            Logger::log(key_name + " = " + print_val, LogLevel::Log_DEBUG);
        }

    } else {
        Logger::log("Corrupted memory map. Expected GGUF.", LogLevel::Log_ERROR);
    }
    //helps prevent memory leak
    UnmapViewOfFile(mapped_data);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    return 0;
}