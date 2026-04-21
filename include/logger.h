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
    ERROR = 0,
    WARN = 1,
    INFO = 2,
    DEBUG = 3
};

class Logger{
    private:
        static LogLevel m_LogLevel;
    public:
        static void setLogLevel(LogLevel LogLevel);
        static void log(std::string_view msg, LogLevel level = LogLevel::INFO);
    
};

#endif