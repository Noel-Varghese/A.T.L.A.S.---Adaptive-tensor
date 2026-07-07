#include <iostream>
#include <string>
#include <windows.h>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../include/logger.h"
#include "../include/memory/arena.h"

//calculaetes exact number of VRAM required
size_t calculate_tensor_size(const std::vector<uint64_t>& shape, uint32_t ggml_type){
    size_t total_elements = 1;
    for(uint64_t dim:shape){
        total_elements *= dim;
    }
    //determining byte size based on ggml id
    //0 = FP32(4bytes), 1 = FP16(2bytes)
    switch(ggml_type){
        case 0:
        return total_elements * 4;//FP32
        case 1:
        return total_elements *2;//FP16
        default:
        return(total_elements * 55)/100;
    }
}


int main(){
    const char* model_path = "models/qwen3-8b-Q4_K_M.gguf";
    
    Logger::log("Booting A.T.L.A.S. ", LogLevel::Log_INFO);
    Logger::log(std::string("Attempting to map: ") + model_path, LogLevel::Log_DEBUG);
    
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        Logger::log("Could not open model file. Check the path.", LogLevel::Log_ERROR);
        return 1;
    }

    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    double size_in_gb = (double)fileSize.QuadPart/(1024*1024*1024);
    Logger::log("Brain Size Detected: " + std::to_string(size_in_gb) + " GB", LogLevel::Log_INFO);

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        Logger::log("Kernel failed to create memory mapping.", LogLevel::Log_ERROR);
        CloseHandle(hFile);
        return 1;
    }

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
    
    if (magic == "GGUF") {
        Logger::log("A.T.L.A.S. is mapped and ready for tensor alignment.", LogLevel::Log_INFO);
        Arena tensor_arena(5368709120ULL);// Spin up a 5GB memory arena (5 * 1024 * 1024 * 1024 bytes)
        // --- PHASE 3 ---
        uint32_t version = *reinterpret_cast<uint32_t*>(data_ptr+4);
        uint64_t tensor_count = *reinterpret_cast<uint64_t*>(data_ptr + 8);
        uint64_t metadata_kv = *reinterpret_cast<uint64_t*>(data_ptr+16);
        
        Logger::log("GGUF Format Version: " + std::to_string(version), LogLevel::Log_DEBUG);
        Logger::log("Neural Network Tensors: " + std::to_string(tensor_count), LogLevel::Log_INFO);
        Logger::log("Hyperparameter KV Pairs: " + std::to_string(metadata_kv), LogLevel::Log_INFO);
        
        // --- PHASE 4: DYNAMIC SCANNER ---
        uint64_t offset = 24;
        Logger::log("Initiating Metadata Scanner", LogLevel::Log_INFO);
        
        for(uint64_t i = 0; i < metadata_kv; ++i) {
            uint64_t key_length = *reinterpret_cast<uint64_t*>(data_ptr + offset);
            offset += 8;
            
            std::string key_name(data_ptr + offset, key_length);
            offset += key_length;

            uint32_t value_type = *reinterpret_cast<uint32_t*>(data_ptr + offset);
            offset += 4;
            
            std::string print_val = "";
            
            if(value_type == 4 || value_type == 5) {
                uint32_t val = *reinterpret_cast<uint32_t*>(data_ptr + offset);
                print_val = std::to_string(val);
                offset += 4;
            } else if(value_type == 6) {
                float val = *reinterpret_cast<float*>(data_ptr + offset);
                print_val = std::to_string(val);
                offset += 4;
            } else if (value_type == 7) { 
                bool val = *reinterpret_cast<bool*>(data_ptr + offset);
                print_val = val ? "true" : "false";
                offset += 1;
            } else if (value_type == 8) { 
                uint64_t str_len = *reinterpret_cast<uint64_t*>(data_ptr + offset);
                offset += 8;
                
                // Extract the full string into memory for the engine
                std::string full_str(data_ptr + offset, str_len);
                offset += str_len;
                
                // FILTER: Only print the first 50 characters to keep the terminal clean
                if (str_len > 50) {
                    print_val = full_str.substr(0, 47) + "... [TRUNCATED FOR TERMINAL]";
                } else {
                    print_val = full_str;
                }
            } else if (value_type == 9) { // ARRAY FAST-FORWARD
                uint32_t arr_type = *reinterpret_cast<uint32_t*>(data_ptr + offset);
                offset += 4;
                
                uint64_t arr_len = *reinterpret_cast<uint64_t*>(data_ptr + offset);
                offset += 8;
                
                print_val = "[ARRAY DETECTED: " + std::to_string(arr_len) + " items - Fast-forwarding pointer]";
                
                for (uint64_t a = 0; a < arr_len; ++a) {
                    if (arr_type == 4 || arr_type == 5 || arr_type == 6) { 
                        offset += 4;
                    } else if (arr_type == 7) { 
                        offset += 1;
                    } else if (arr_type == 8) { 
                        uint64_t str_len = *reinterpret_cast<uint64_t*>(data_ptr + offset);
                        offset += 8 + str_len; 
                    }
                }
            } else {
                print_val = "[UNKNOWN TYPE ID: " + std::to_string(value_type) + "]";
                Logger::log(key_name + " = " + print_val, LogLevel::Log_ERROR);
                break; 
            }   
            Logger::log(key_name + " = " + print_val, LogLevel::Log_DEBUG);
        } 

        // --- PHASE 5: TENSOR ROSTER ---
        Logger::log("--------------INITIATING TENSOR ROSTER SCAN ---------------", LogLevel::Log_INFO);
        Logger::log("Starting loop for " + std::to_string(tensor_count) + " tensors. Offset: " + std::to_string(offset), LogLevel::Log_INFO);
        struct TensorInfo
        {
            std::string name;
            std::vector<uint64_t> shape;
            uint32_t ggml_type;
            uint64_t rel_offset;
            std::string shape_str;
        };
        std::vector<TensorInfo> tensor_infos;
        tensor_infos.reserve(tensor_count);
        
        for(uint64_t i = 0; i < tensor_count; ++i) {
            if (i > 0 && i % (tensor_count / 4) == 0) {
                Logger::log("Hydration Progress: " + std::to_string((i * 100) / tensor_count) + "%", LogLevel::Log_INFO);
            }
            Logger::log("Processing Tensor [" + std::to_string(i) + "] at Offset: " + std::to_string(offset), LogLevel::Log_DEBUG);
            uint64_t name_len = *reinterpret_cast<uint64_t*>(data_ptr + offset);
            offset += 8;
            
            std::string tensor_name(data_ptr + offset, name_len);
            offset += name_len;
            
            uint32_t n_dims = *reinterpret_cast<uint32_t*>(data_ptr + offset);
            offset += 4;
            
            std::string shape_str = "[";
            std::vector<uint64_t> tensor_shape;
            for (uint32_t d = 0; d < n_dims; ++d) {
                uint64_t dim_size = *reinterpret_cast<uint64_t*>(data_ptr + offset);
                tensor_shape.push_back(dim_size);
                shape_str += std::to_string(dim_size);
                if (d < n_dims - 1) shape_str += ", ";
                offset += 8;
            }
            shape_str += "]";
            
            uint32_t ggml_type = *reinterpret_cast<uint32_t*>(data_ptr + offset);
            offset += 4;

            uint64_t tensor_offset = *reinterpret_cast<uint64_t*>(data_ptr + offset);
            offset += 8;
            
            if(i < 5 || i == tensor_count - 1) {
                Logger::log("Tensor [" + std::to_string(i) + "]: " + tensor_name + 
                            " | Shape: " + shape_str + 
                            " | Type ID: " + std::to_string(ggml_type), LogLevel::Log_DEBUG);
            }
            if(i == 5) {
                Logger::log("... [393 Tensors Hidden for Terminal Safety] ...", LogLevel::Log_INFO);
            }
           tensor_infos.push_back({std::move(tensor_name), std::move(tensor_shape), ggml_type, tensor_offset, shape_str});

        }
        uint64_t tensor_data_section_strt = (offset+31) & ~31ULL;
        Logger::log("Tensor data section begins at: " + std::to_string(tensor_data_section_strt), LogLevel::Log_INFO);
        for(uint64_t i=0;i<tensor_infos.size();++i){
            const TensorInfo& t = tensor_infos[i];
            if(i>0&& i % (tensor_infos.size()/4) == 0){
                Logger::log("Hydration Progress: " + std::to_string((i * 100) / tensor_infos.size()) + "%", LogLevel::Log_INFO);
            }
            size_t current_tensor_byte_size = calculate_tensor_size(t.shape, t.ggml_type);
            void* aligned_tensor_memory = data_ptr + tensor_data_section_strt + t.rel_offset;
            // Logger::log("Dest Pointer: " + std::to_string(reinterpret_cast<uint64_t>(aligned_tensor_memory)), LogLevel::Log_INFO);
            // Logger::log("Source Pointer: " + std::to_string(reinterpret_cast<uint64_t>(data_ptr + tensor_data_section_strt + t.rel_offset)), LogLevel::Log_INFO);
            // std::memcpy(aligned_tensor_memory, data_ptr+tensor_data_section_strt+t.rel_offset,current_tensor_byte_size);
            if(i<5 || i == tensor_infos.size()-1){
                uint8_t* byte_ptr = static_cast<uint8_t*>(aligned_tensor_memory);
                Logger::log("Hydration Check [Byte 0]: " + std::to_string(byte_ptr[0]), LogLevel::Log_DEBUG);
            }
        }
        Logger::log("Engine successfully mapped all 399 architectural boundaries.", LogLevel::Log_INFO);

    } else {
        Logger::log("Corrupted memory map. Expected GGUF.", LogLevel::Log_ERROR);
    }
    
    UnmapViewOfFile(mapped_data);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    return 0;
}
