#pragma once
#include <cuda_runtime.h>
#include <chrono>

class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

class CPUTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};
