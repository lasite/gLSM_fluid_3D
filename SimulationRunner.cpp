#include "SimulationRunner.h"

#include "gelSystem.h"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {

std::filesystem::path resolve_output_root(const std::filesystem::path& desiredRoot,
                                          const std::filesystem::path& originalDirectory) {
    if (desiredRoot.empty()) {
        return originalDirectory;
    }

    if (desiredRoot.is_absolute()) {
        return desiredRoot;
    }

    return originalDirectory / desiredRoot;
}

}  // namespace

SimulationRunner::ScopedCurrentPath::ScopedCurrentPath(const std::filesystem::path& newPath)
    : previousPath_(std::filesystem::current_path()) {
    std::filesystem::current_path(newPath);
}

SimulationRunner::ScopedCurrentPath::~ScopedCurrentPath() {
    std::error_code ec;
    std::filesystem::current_path(previousPath_, ec);
    if (ec) {
        std::cerr << "Warning: failed to restore working directory: " << ec.message() << std::endl;
    }
}

SimulationRunner::SimulationRunner(SimulationConfig config)
    : config_(std::move(config)), originalDirectory_(std::filesystem::current_path()) {
    if (config_.iValues.empty() || config_.jValues.empty() || config_.kValues.empty()) {
        throw std::invalid_argument("Simulation parameter sets must not be empty");
    }

    config_.outputRoot = resolve_output_root(config_.outputRoot, originalDirectory_);
}

void SimulationRunner::run() {
    const auto cudaStatus = cudaSetDevice(config_.deviceId);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to select CUDA device: ") + cudaGetErrorString(cudaStatus));
    }

    std::filesystem::create_directories(config_.outputRoot);

    std::size_t count = 0;
    for (int iValue : config_.iValues) {
        for (int jValue : config_.jValues) {
            for (int kValue : config_.kValues) {
                runCombination(count, iValue, jValue, kValue);
                ++count;
            }
        }
    }

    std::filesystem::current_path(originalDirectory_);
}

void SimulationRunner::runCombination(std::size_t count, int iValue, int jValue, int kValue) {
    const auto runDirectory = config_.outputRoot / std::to_string(count);
    std::filesystem::create_directories(runDirectory);

    ScopedCurrentPath directoryGuard(runDirectory);

    auto gel = std::make_unique<GelSystem>(config_.gelSize, config_.fluidSize, config_.time, iValue, jValue, kValue);
    gel->m_count = static_cast<int>(count);

    std::cout << count << ", ";
    std::cout << "CHS = " << gel->m_params.CHS << ", f = " << gel->m_params.f
              << ", ep = " << gel->m_params.ep << ", I = " << gel->m_params.I << std::endl;

    const auto startIteration = static_cast<long long>(config_.time) * gel->m_df;
    for (long long solverIterations = startIteration; solverIterations <= config_.runStep; ++solverIterations) {
        if (gel->result) {
            gel->update(solverIterations);
        } else {
            break;
        }
    }
}
