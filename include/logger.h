#ifndef __LOGGER_H_
#define __LOGGER_H_

#include<iostream>
#include<string_view>

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define GREEN   "\033[32m"
#define CYAN    "\033[36m"

enum class LogLevel{
    Log_ERROR = 0,
    Log_WARN = 1,
    Log_INFO = 2,
    Log_DEBUG = 3
};

class Logger{
    private:
        static LogLevel m_LogLevel;
    public:
        static void setLogLevel(LogLevel LogLevel);
        static void log(std::string_view msg, LogLevel level = LogLevel::Log_INFO);
    
};

#endif