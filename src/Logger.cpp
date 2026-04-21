#include "../include/logger.h"

LogLevel Logger::m_LogLevel = LogLevel::DEBUG;

void Logger::setLogLevel(LogLevel level) {
    m_LogLevel = level;
}

void Logger::log(std::string_view msg, LogLevel level) {
    if (level <= m_LogLevel) {
        switch (level) {
            case LogLevel::ERROR:
                std::cout << RED << "[ERROR] " << msg << RESET << "\n";
                break;
            case LogLevel::WARN:
                std::cout << YELLOW << "[WARN] " << msg << RESET << "\n";
                break;
            case LogLevel::INFO:
                std::cout << GREEN << "[INFO] " << msg << RESET << "\n";
                break;
            case LogLevel::DEBUG:
                std::cout << CYAN << "[DEBUG] " << msg << RESET << "\n";
                break;
        }
    }
}