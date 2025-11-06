#include "gelSystem.h"
#include <direct.h>
#include <string>
#include <iostream>

using namespace std;

#define runstep 1000000

GelSystem* gel = 0;

int main(int argc, char** argv) {
    cudaSetDevice(1);
    int time = 0;
    int count = 0;
    char originalDirectory[FILENAME_MAX];

    if (!_getcwd(originalDirectory, sizeof(originalDirectory))) {
        cerr << "Error getting current directory" << endl;
        return EXIT_FAILURE;
    }

    for (int i = 9; i <= 9; i++) {
        for (int j = 1; j <= 1; j++) {
            for (int k = 0; k <= 0; k++) {
                string directoryName = to_string(count);
                _mkdir(directoryName.c_str());
                _chdir(directoryName.c_str());

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

                delete gel;
                _chdir(originalDirectory);
                count++;
            }
        }
    }

    return 0;
}
