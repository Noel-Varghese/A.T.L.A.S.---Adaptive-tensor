#include "../include/logger.h"

LogLevel Logger::m_LogLevel = LogLevel::Log_DEBUG;

void Logger::setLogLevel(LogLevel level) {
    m_LogLevel = level;
}

void Logger::log(std::string_view msg, LogLevel level) {
    if (level <= m_LogLevel) {
        switch (level) {
            case LogLevel::Log_ERROR:
                std::cout << RED << "[ERROR] " << msg << RESET << "\n";
                break;
            case LogLevel::Log_WARN:
                std::cout << YELLOW << "[WARN] " << msg << RESET << "\n";
                break;
            case LogLevel::Log_INFO:
                std::cout << GREEN << "[INFO] " << msg << RESET << "\n";
                break;
            case LogLevel::Log_DEBUG:
                std::cout << CYAN << "[DEBUG] " << msg << RESET << "\n";
                break;
        }
    }
}