#pragma once
#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count()/1000.0;
        std::cout << name_ << " cost: " << duration << " ms " << std::endl;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
#endif // !TIMER_HPP
