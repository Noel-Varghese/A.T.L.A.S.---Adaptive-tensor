#include <iostream>
#include <fstream>
#include "../include/logger.h"

int main(){
    const char* model_path = "models/Qwen3-8B-Q4_K_M.gguf";
    Logger::log("Booting A.T.L.A.S. ", LogLevel::INFO);
    Logger::log(std::string("attempting to open: ")+model_path, LogLevel::DEBUG);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        Logger::log("Could not open model file. Check the path.", LogLevel::ERROR);
        return 1;
    }
    char magic[5] = {0};//reads the first four bits
    file.read(magic, 4);
    Logger::log(std::string("First 4 bytes read: [") + magic + "]", LogLevel::DEBUG);
    if (std::string(magic) == "GGUF") {
        Logger::log("Magic Number verified. The Brain is connected.", LogLevel::INFO);
    } else {
        Logger::log("Invalid file format. Expected GGUF.", LogLevel::ERROR);
    }

    file.close();
    return 0;
}