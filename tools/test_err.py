import re
with open("sim_gelonly_tube.cpp", "r") as f:
    code = f.read()
new_code = code.replace('cudaDeviceSynchronize();',
'''cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\\n", cudaGetErrorString(err));
            }''')
with open("sim_gelonly_tube.cpp", "w") as f:
    f.write(new_code)
