#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <filesystem>
#include <vector>

struct SimulationConfig {
    int deviceId = 0;
    int time = 0;
    long long runStep = 1'000'000;
    int3 gelSize{};
    int3 fluidSize{};
    std::vector<int> iValues;
    std::vector<int> jValues;
    std::vector<int> kValues;
    std::filesystem::path outputRoot;
};

class SimulationRunner {
public:
    explicit SimulationRunner(SimulationConfig config);

    void run();

private:
    void runCombination(std::size_t count, int iValue, int jValue, int kValue);

    class ScopedCurrentPath {
    public:
        explicit ScopedCurrentPath(const std::filesystem::path& newPath);
        ~ScopedCurrentPath();

    private:
        std::filesystem::path previousPath_;
    };

    SimulationConfig config_;
    std::filesystem::path originalDirectory_;
};
