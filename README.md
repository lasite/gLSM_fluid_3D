# gLSM Fluid 3D Project Documentation

This document provides an in-depth reference for contributors and advanced users of the **gLSM Fluid 3D** CUDA code base. It explains how the solver couples an active gel model to a three-dimensional lattice Boltzmann (LBM) fluid, how the repository is organized, what runtime assets are generated, and which implementation gaps still require attention.

---

## Table of Contents
1. [Scientific Overview](#scientific-overview)
2. [Codebase Layout](#codebase-layout)
3. [Build & Toolchain Guidance](#build--toolchain-guidance)
4. [Execution Pipeline](#execution-pipeline)
5. [Runtime Configuration](#runtime-configuration)
6. [Key Data Structures](#key-data-structures)
7. [Output Artefacts](#output-artefacts)
8. [Extending the Solver](#extending-the-solver)
9. [Implementation Caveats](#implementation-caveats)
10. [Troubleshooting & Profiling Tips](#troubleshooting--profiling-tips)

---

## Scientific Overview
The simulation resolves a deformable, chemically active gel immersed in a viscous fluid. The gel is modeled on a Lagrangian mesh whose nodes store:

- **Geometry** – Positions (`rn`), element centers (`rm`), and director information.
- **Chemistry** – Concentrations (`um`, `vm`, `wm`) that drive internal contractility.
- **Mechanics** – Velocity (`Veln`), deformation gradient (`Fn`), and reaction force (`Fn`).

The surrounding fluid is advanced on an Eulerian lattice via an LBM collide/stream routine. Gel–fluid coupling is implemented with an immersed boundary method (IBM): gel boundary forces are spread to the fluid grid, fluid velocities are interpolated back to the gel nodes, and both subsystems iterate until the next macro time step.

---

## Codebase Layout
| File / Directory | Description |
| ---------------- | ----------- |
| `gel.cpp` | Program entry point. Configures parameter sweeps, prepares output directories, launches `GelSystem` updates, and manages checkpoint/restart files. |
| `gelSystem.h` / `gelSystem.cu` | Host-side orchestration of the coupled gel–fluid solver. Handles memory allocation on CPU/GPU, maintains simulation clocks, triggers CUDA kernels, and streams diagnostic output. |
| `gel_kernel.cu` / `gel_kernel.cuh` | CUDA kernel implementations and declarations. Includes routines for chemistry diffusion/reaction, mechanical updates, IBM interpolation/spreading, and the LBM collide/stream steps. |
| `gelParams.h` | Parameter structures shared between host and device code. Encapsulates gel material constants, IBM coefficients, lattice geometry, and numerical control flags. |
| `gLSM_fluid_3D.sln` / `gLSM_fluid_3D.vcxproj` | Visual Studio + CUDA project for building on Windows systems. |

The repository currently ships only source and project files; tests and post-processing scripts are not included.

---

## Build & Toolchain Guidance
### Windows (Visual Studio)
1. Install the NVIDIA CUDA Toolkit (version 10.2 or higher recommended) alongside Visual Studio 2019 or newer.
2. Open `gLSM_fluid_3D.sln`, ensure the CUDA build customization is enabled, and select a GPU architecture (e.g., `sm_75`).
3. Build the `Release` configuration. Output binaries are placed next to the solution file.

### Linux (command-line)
A CMake file is not provided, but an equivalent build can be scripted using `nvcc`:
```bash
nvcc -O3 -std=c++17 -arch=sm_70 \
     gel.cpp gelSystem.cu gel_kernel.cu \
     -o gLSM_fluid_3D
```
Ensure `CUDA_HOME` is set and the host compiler supports C++17. You may need to add `-Xcompiler -fopenmp` if using OpenMP features (currently unused by the code).

### Common Requirements
- CUDA-capable GPU with sufficient device memory to store gel + fluid arrays (hundreds of MB for typical grids).
- Adequate CPU RAM for mirrored host buffers and asynchronous file output.
- Optional: Nsight Compute / Nsight Systems for profiling, and Python/Matlab for post-processing `.dat` outputs.

---

## Execution Pipeline
The high-level simulation loop mirrors the following sequence:

1. **Initialization**
   - `GelSystem` constructor records grid sizes, time step counts, and sweep indices supplied by `gel.cpp`.
   - `_initialize` allocates host/device buffers, zeros or seeds them, and uploads parameter structs to constant memory.
   - Initial gel geometry and chemistry fields are populated, typically from analytic formulas embedded in the code.

2. **Main Time Step (`GelSystem::update`)**
   - *Chemistry:* `calChemD` advances reaction–diffusion equations on the gel mesh.
   - *Geometry & Mechanics:* Kernels such as `calNodesGeometryD`, `calNodesVelocityD`, and `calNodesForceD` refresh deformation metrics and nodal forces.
   - *Immersed Boundary Sub-iterations:* For `Nsub` cycles the code
     1. Packs Lagrangian boundary data into `d_lag`/`d_Vl`.
     2. Interpolates Eulerian velocities to the gel nodes (`k_ibm_interp`).
     3. Updates gel mechanics with the interpolated velocities and computes new forces.
     4. Spreads forces back to the fluid grid (`k_ibm_spread`).
   - *Fluid Step:* Executes the LBM collide (`k_collide`) and stream/bounce-back (`k_stream_bounce`) kernels, using the IBM force density as an external input.
   - *Diagnostics:* On configurable intervals, data are copied to host pinned buffers and a writer thread dumps `.dat` files.

3. **Finalization**
   - Once the requested number of steps (`runstep`) is reached, CUDA resources are freed, file handles closed, and statistics printed.

---

## Runtime Configuration
Current configuration is hard-coded inside `gel.cpp` and `gelParams.h`. Important knobs include:

- **Grid Dimensions:** `Nx`, `Ny`, `Nz` define the fluid lattice. Gel discretization counts (`m_Numfilaments`, `m_NumNodes`) are derived from parameter structs.
- **Time Control:** `runstep`, `Nsub`, and `dt` determine the macro time step length and number of immersed boundary subcycles.
- **Parameter Sweeps:** The main program iterates `multiRun`, `multiB`, `multiC`, etc., changing material constants between runs and writing to distinct directories.
- **Restart Flags:** `setGoonValue` attempts to resume from checkpoint files if `goon` is enabled.

To externalize configuration, consider reading JSON or INI files and populating `params`/`p` structures prior to constructing `GelSystem`.

---

## Key Data Structures
### Gel-related
- `m_hum`, `m_hvm`, `m_hwm` (host) and `m_dum`, `m_dvm`, `m_dwm` (device) store per-node chemistry fields.
- `m_hrn`, `m_hVeln`, `m_hFn` and their device counterparts hold geometry, velocity, and force vectors.
- `m_hfilament` and `m_dfilament` represent filament configurations used to compute active stresses.

### Fluid-related
- `d_f`, `d_fpost`, `d_fnext`: LBM distribution arrays for streaming and collision steps.
- `d_rho`, `d_u`: Macroscopic density and velocity fields.
- `d_F_ibm`: Accumulated body force contributions from the gel.

### Coupling Buffers
- `d_lag`, `d_Ul`, `d_Vl`: Lagrangian data exchanged between gel and fluid during IBM iterations.
- `d_Fl`: Gel reaction forces awaiting distribution to the fluid grid.
- `d_A`: Intended auxiliary array for IBM spreading (currently not allocated—see [Implementation Caveats](#implementation-caveats)).

---

## Output Artefacts
Each parameter sweep run creates a numbered directory (`0`, `1`, …). Within it the solver periodically writes ASCII files of the form:

| Filename Pattern | Contents |
| ---------------- | -------- |
| `rnXXXX.dat`, `rmXXXX.dat` | Gel node and element center positions at time index `XXXX`.
| `umXXXX.dat`, `vmXXXX.dat`, `wmXXXX.dat` | Chemistry fields `u`, `v`, `w` sampled over the gel mesh.
| `FnXXXX.dat` | Gel nodal force vectors.
| `VelnXXXX.dat` | Gel nodal velocities.
| `VelbXXXX.dat` | Eulerian fluid velocity field.

The files can be ingested by Python/Matlab scripts or converted to VTK for visualization.

---

## Extending the Solver
1. **Multiple Gels:** Introduce a `GelInstance` structure and teach `GelSystem`/kernels to iterate over multiple instances with separate offsets to study gel–gel interactions.
2. **Portable Runtime:** Replace Windows-specific `_mkdir`/`_chdir` calls with `std::filesystem` and make the CUDA device selection configurable instead of hard-coding `cudaSetDevice(1)`.
3. **Improved Restart Handling:** Validate checkpoint file availability before loading, and allow the restart directory to be configured.
4. **Physics Enhancements:** Add new reaction kinetics, active stress models, or boundary conditions by augmenting CUDA kernels and associated parameter packs.
5. **Testing Infrastructure:** Create regression tests that run a few time steps on a small grid, verifying conservation laws and buffer initialization (e.g., ensuring `m_hfilament` is zeroed correctly).

---

## Implementation Caveats
The current code base contains several issues highlighted during inspection:

- `cudaSetDevice(1)` in `gel.cpp` assumes a second GPU and will fail on single-GPU systems; it should query available devices and fall back to `0`.
- Directory management in `gel.cpp` uses Windows-only APIs (`_mkdir`, `_chdir`), hindering portability.
- `allocateHostStorage` clears `m_hfilament` with `sizeof(bool)` instead of `sizeof(double3)`, leaving part of the buffer uninitialized.
- `d_A` is passed to IBM kernels but never allocated in `allocateDeviceStorage`, risking undefined behavior.
- `setGoonValue` does not verify checkpoint file availability, so restarts silently fail when files are missing.

Addressing these items is recommended before large-scale studies.

---

## Troubleshooting & Profiling Tips
- **Runtime Failures:** Enable CUDA error checking after every API call (`cudaGetLastError`) to detect invalid allocations or launches early.
- **Performance:** Use Nsight Compute to confirm memory bandwidth usage of key kernels (`k_ibm_spread`, `k_collide`). Ensure coalesced memory access by aligning gel node counts with warp sizes where possible.
- **I/O Bottlenecks:** The asynchronous writer thread can lag behind on large grids. Consider batching outputs or writing in binary to reduce file size.
- **Portability:** When porting to Linux, substitute `_mkdir`/`_chdir` with `std::filesystem::create_directories` / `current_path` and guard platform-specific includes with `#ifdef _WIN32`.

---

For further questions or contributions, please open an issue or submit a pull request describing your proposed enhancements.
