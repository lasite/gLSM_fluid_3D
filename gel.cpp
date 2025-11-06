#include "SimulationRunner.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

SimulationConfig build_default_config() {
    SimulationConfig config;
    config.deviceId = 1;
    config.time = 0;
    config.runStep = 1'000'000;
    config.gelSize = make_int3(21, 21, 21);
    config.fluidSize = make_int3(101, 101, 101);
    config.iValues = {9};
    config.jValues = {1};
    config.kValues = {0};
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        SimulationRunner runner(build_default_config());
        runner.run();
    } catch (const std::exception& ex) {
        std::cerr << "Simulation failed: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
