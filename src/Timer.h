#pragma once
#include <iostream>
#include <chrono>
#include <string>

#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_RESET   "\x1b[0m"

class Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
    std::string name = "";

public:
    Timer()
    {
        m_StartTimePoint = std::chrono::high_resolution_clock::now();
    }
    Timer(std::string str)
    {
        m_StartTimePoint = std::chrono::high_resolution_clock::now();
        name = str;
    }

    ~Timer()
    {
        stop();
    }

    void stop()
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> endTimePoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;
        std::cout << ANSI_COLOR_BLUE << name << " <- ";
        std::cout << duration << "us (" << ms << "ms)\n" ANSI_COLOR_RESET;
    }
};