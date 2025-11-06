#include "gelSystem.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

using namespace std;

#define runstep 1000000

GelSystem* gel = 0;

int main(int argc, char** argv) {
    int deviceIndex = 0;
    if (argc > 1) {
        try {
            deviceIndex = stoi(argv[1]);
        }
        catch (const invalid_argument&) {
            cerr << "Invalid device index '" << argv[1] << "', expected integer." << endl;
            return EXIT_FAILURE;
        }
        catch (const out_of_range&) {
            cerr << "Device index out of range: " << argv[1] << endl;
            return EXIT_FAILURE;
        }
    }

    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        cerr << "Failed to query CUDA devices: " << cudaGetErrorString(cudaStatus) << endl;
        return EXIT_FAILURE;
    }
    if (deviceCount == 0) {
        cerr << "No CUDA-capable devices available." << endl;
        return EXIT_FAILURE;
    }
    if (deviceIndex < 0 || deviceIndex >= deviceCount) {
        cerr << "Requested device " << deviceIndex << " unavailable; falling back to device 0." << endl;
        deviceIndex = 0;
    }
    cudaStatus = cudaSetDevice(deviceIndex);
    if (cudaStatus != cudaSuccess) {
        cerr << "Failed to set CUDA device " << deviceIndex << ": " << cudaGetErrorString(cudaStatus) << endl;
        return EXIT_FAILURE;
    }

    namespace fs = std::filesystem;

    fs::path originalDirectory;
    try {
        originalDirectory = fs::current_path();
    }
    catch (const fs::filesystem_error& error) {
        cerr << "Error getting current directory: " << error.what() << endl;
        return EXIT_FAILURE;
    }

    int time = 0;
    int count = 0;

    for (int i = 9; i <= 9; i++) {
        for (int j = 1; j <= 1; j++) {
            for (int k = 0; k <= 0; k++) {
                const fs::path runDirectory = originalDirectory / to_string(count);
                std::error_code ec;
                if (!fs::exists(runDirectory)) {
                    fs::create_directory(runDirectory, ec);
                    if (ec) {
                        cerr << "Failed to create directory '" << runDirectory.string()
                             << "': " << ec.message() << endl;
                        continue;
                    }
                }

                fs::current_path(runDirectory, ec);
                if (ec) {
                    cerr << "Failed to enter directory '" << runDirectory.string()
                         << "': " << ec.message() << endl;
                    continue;
                }

                try {
                    int3 gelSize = make_int3(21, 21, 21);
                    int3 fluidSize = make_int3(101, 101, 101);
                    gel = new GelSystem(gelSize, fluidSize, time, i, j, k);
                    gel->m_count = count;
                    cout << count << ", ";
                    cout << "CHS = " << gel->m_params.CHS << ", f = " << gel->m_params.f
                        << ", ep = " << gel->m_params.ep << ", I = " << gel->m_params.I
                        << endl;

                    for (long long int solverIterations = static_cast<long long int>(time) * gel->m_df; solverIterations <= runstep; solverIterations++) {
                        if (gel->result) {
                            gel->update(solverIterations);
                        }
                        else {
                            break;
                        }
                    }
                }
                catch (const exception& ex) {
                    cerr << "Simulation failed in directory '" << runDirectory.string() << "': "
                         << ex.what() << endl;
                }

                delete gel;
                gel = nullptr;
                fs::current_path(originalDirectory, ec);
                if (ec) {
                    cerr << "Failed to restore working directory to '" << originalDirectory.string()
                         << "': " << ec.message() << endl;
                    return EXIT_FAILURE;
                }
                count++;
            }
        }
    }

    return 0;
}
