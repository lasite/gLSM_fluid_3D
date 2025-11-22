# Build Instructions for gLSM_fluid_3D

This project is a **Visual Studio 2022** project that requires **NVIDIA CUDA Toolkit 12.8**.

## Prerequisites

1.  **Visual Studio 2022** with "Desktop development with C++" workload installed.
2.  **NVIDIA CUDA Toolkit 12.8**.
    *   Ensure that the CUDA Visual Studio integration is installed.
3.  **NVIDIA GPU** supporting Compute Capability 8.6 (for Release) or 8.9 (for Debug).
    *   *Note: If you have a different GPU, you may need to adjust the `CodeGeneration` setting in the project properties.*

## Compilation Methods

### Option 1: Using Visual Studio IDE

1.  Open the `gLSM_fluid_3D.sln` file in Visual Studio 2022.
2.  Select the desired configuration (e.g., `Release` or `Debug`) and platform (`x64`) from the toolbar.
3.  Right-click on the project `gLSM_fluid_3D` in the Solution Explorer and select **Build**.
4.  The executable will be generated in the `x64/Release` or `x64/Debug` directory.

### Option 2: Using Developer Command Prompt

1.  Open the **Developer Command Prompt for VS 2022**.
2.  Navigate to the project directory:
    ```cmd
    cd d:\repos\gLSM_fluid_3D
    ```
3.  Run `msbuild` to build the project:
    ```cmd
    msbuild gLSM_fluid_3D.sln /p:Configuration=Release /p:Platform=x64
    ```

## Troubleshooting

*   **CUDA Version Mismatch**: If you have a different version of CUDA installed, you can edit `gLSM_fluid_3D.vcxproj` and change `<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />` and `<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.targets" />` to match your installed version (e.g., `CUDA 12.x`).
*   **Compute Capability**: If you get an error regarding "sm_86" or "sm_89" not being supported, go to **Project Properties > CUDA C/C++ > Device > Code Generation** and update it to match your GPU's compute capability (e.g., `compute_75,sm_75` for RTX 20 series).
