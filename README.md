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
| `sim.cpp` | Program entry point that instantiates two gels, a `Coupler`, and the surrounding `Fluid`. Runs the top-level time-stepping loop that alternates gel mechanics/chemistry updates, immersed boundary packing/scattering, and fluid LBM steps. |
| `gel.h` / `gel.cu` | Gel data model and update routines. Manages host/device buffers for nodal positions, forces, chemistry fields, and boundary indices; launches CUDA kernels defined in `gel_kernels.cu[h]`; and writes gel-specific outputs (`gel<id>rnXXXX.dat`, etc.). |
| `fluid.h` / `fluid.cu` | LBM-based fluid simulator. Owns distribution functions, macroscopic fields, and immersed boundary force buffers; advances velocity and concentration; and writes `VelbXXXX.dat` / `ConcXXXX.dat` snapshots. |
| `coupling.h` / `coupling.cu` | Immersed boundary glue between gels and fluid. Builds packed Lagrangian data, accumulates force/velocity exchanges, and exposes GPU buffers to both subsystems. |
| `*_kernels.cu` / `*_kernels.cuh` | CUDA kernels and declarations for gel mechanics/chemistry, fluid collide/stream, and coupling interpolation/spreading operations. |
| `gLSM_fluid_3D.sln` / `gLSM_fluid_3D.vcxproj` | Visual Studio + CUDA project files for Windows builds. |

The repository currently ships only source and project files; tests and post-processing scripts are not included.

---

## Build & Toolchain Guidance
### Windows (Visual Studio)
1. Install the NVIDIA CUDA Toolkit (version 10.2 or higher recommended) alongside Visual Studio 2019 or newer.
2. Open `gLSM_fluid_3D.sln`, ensure the CUDA build customization is enabled, and select a GPU architecture (e.g., `sm_75`).
3. Build the `Release` configuration. Output binaries are placed next to the solution file.

### Linux (command-line)
A CMake file is not provided. You can build the standalone executable with `nvcc` by compiling the entry point, subsystem sources, and their kernels:
```bash
nvcc -O3 -std=c++17 -arch=sm_70 \
     sim.cpp gel.cu fluid.cu coupling.cu \
     gel_kernels.cu fluid_kernels.cu coupling_kernels.cu \
     -o gLSM_fluid_3D
```
If you see errors about `cudaSetDevice(1)`, switch the hard-coded device in `sim.cpp` to `0` or any available device ID. Ensure `CUDA_HOME` is set and the host compiler supports C++17.

### Common Requirements
- CUDA-capable GPU with sufficient device memory to store gel + fluid arrays (hundreds of MB for typical grids).
- Adequate CPU RAM for mirrored host buffers and asynchronous file output.
- Optional: Nsight Compute / Nsight Systems for profiling, and Python/Matlab for post-processing `.dat` outputs.

---

## Execution Pipeline
The high-level simulation loop in `sim.cpp` mirrors the following sequence:

1. **Initialization**
   - Two `Gel` instances are built with user-chosen sizes, positions, and types.
   - A `Coupler` is created to aggregate gel boundary data and expose buffers to the fluid.
   - A `Fluid` object initializes the LBM lattice, distribution functions, and immersed boundary force arrays.

2. **Main Time Step (`for` loop in `main`)**
   - *Gel updates:* Each gel calls `stepElasticity` then `stepChemistry`, advancing mechanical and reaction–diffusion fields on its Lagrangian mesh.
   - *Coupling pack:* `Coupler::packFromGels` collects boundary positions/forces into contiguous device buffers.
   - *Fluid step:* `Fluid::stepVelocity` and `Fluid::stepConcentration` execute collide/stream and scalar advection/diffusion substeps, incorporating IBM forces.
   - *Coupling scatter:* `Coupler::scatterToGels` interpolates updated fluid velocities back to the gel boundaries.
   - *Diagnostics:* Every 1000 iterations, gels and fluid copy data to the host and spawn writer threads to dump `.dat` snapshots.

3. **Finalization**
   - After the configured `runstep` iterations, host/device resources are freed and writer threads are joined.

---

## Runtime Configuration
Current configuration is hard-coded inside `sim.cpp`, `gel.cu`, and `fluid.cu`. Important knobs include:

- **Grid Dimensions:** `fluidSize` in `sim.cpp` controls the Eulerian lattice; `gelSize` and `gelPosition` determine each gel’s discretization and placement.
- **Time Control:** `runstep` in `sim.cpp` sets the number of macro iterations. Gel and fluid time steps (`m_dt` and `dt`) are fixed to `1e-3`, and the fluid internally computes `Nsub` immersed boundary substeps from its viscosity.
- **Material Parameters:** `GelParams` and `FluidParams` structs (allocated in constructors) store reaction kinetics, lattice weights, and diffusion coefficients. They are currently initialized with constants in code.
- **Device Selection:** `cudaSetDevice(1)` in `sim.cpp` assumes a second GPU; change to `0` if only one device is present.

To externalize configuration, consider reading JSON or INI files and populating these parameters before constructing `Gel`, `Coupler`, and `Fluid` instances.

---

## Key Data Structures
### Gel-related
- `m_hum`, `m_hvm`, `m_hwm` (host) and `m_dum`, `m_dvm`, `m_dwm` (device) store per-element chemistry fields.
- `m_hrn`, `m_hVeln`, `m_hFn` and their device counterparts hold nodal geometry, velocity, and force vectors; `m_hrm`/`m_drm` capture element centers.
- `m_hmap_node`, `m_hmap_element`, and `m_hbIndex` (plus GPU mirrors) encode boundary node/element indices and neighbor lookups used by the immersed boundary scheme.
- `m_hgp` / `m_dgp` hold `GelParams`, including geometry offsets and reaction/mechanical coefficients.

### Fluid-related
- `d_f`, `d_fpost`, `d_fnext`: LBM distribution arrays for streaming and collision steps.
- `d_rho`, `d_u`: Macroscopic density and velocity fields.
- `d_F_ibm`: Accumulated body force contributions from the gel.
- `d_c1`, `d_c2`: Scalar concentration fields advanced alongside velocity.

### Coupling Buffers
- `d_lag_all_`, `d_Ul_all_`, `d_Vl_all_`: Aggregated Lagrangian boundary positions and velocities for all gels.
- `d_Fl_all_`, `d_Dl_all_`: Aggregated boundary forces and marker spacings used when spreading to the Eulerian grid.
- `d_owner`: Gel ownership flags for each packed boundary entry.
- `d_A`: Intended auxiliary array for IBM spreading (currently not allocated—see [Implementation Caveats](#implementation-caveats)).

---

## Output Artefacts
All outputs are written to the working directory every 1000 iterations (time is reported in physical units using `dt`). Filenames encode the gel ID and iteration time:

| Filename Pattern | Contents |
| ---------------- | -------- |
| `gel<id>rnXXXX.dat`, `gel<id>rmXXXX.dat` | Gel node (`rn`) and element center (`rm`) positions for gel `<id>` at time `XXXX`. |
| `gel<id>umXXXX.dat`, `gel<id>vmXXXX.dat`, `gel<id>wmXXXX.dat` | Chemistry fields `u`, `v`, `w` on the gel elements. |
| `gel<id>FnXXXX.dat`, `gel<id>VelnXXXX.dat` | Gel nodal forces and velocities. |
| `VelbXXXX.dat` | Eulerian fluid velocity field. |
| `ConcXXXX.dat` | Fluid concentration scalar field. |

The files can be ingested by Python/Matlab scripts or converted to VTK for visualization.

---

## Extending the Solver
1. **Configurable Scenes:** Replace the hard-coded two-gel setup in `sim.cpp` with file- or CLI-driven scene descriptions so arbitrary gel counts, positions, and material types can be simulated without recompiling.
2. **Portable Runtime:** Expose GPU selection and output locations via arguments or configuration files instead of relying on in-source constants such as `cudaSetDevice(1)`.
3. **Checkpoint/Restart:** Add periodic checkpoint writing and a restart path to avoid losing long simulations if the process exits unexpectedly.
4. **Physics Enhancements:** Add new reaction kinetics, active stress models, or boundary conditions by augmenting CUDA kernels and associated parameter packs.
5. **Testing Infrastructure:** Create regression tests that run a few time steps on a small grid, verifying conservation laws and buffer initialization before larger runs.

---

## Implementation Caveats
The current code base contains several issues highlighted during inspection:

- `cudaSetDevice(1)` in `sim.cpp` assumes a second GPU and will fail on single-GPU systems; it should query available devices and fall back to `0`.
- `d_A` is passed to IBM kernels in `fluid.cu` but never allocated in `allocateDeviceStorage`, risking undefined behavior.
- CUDA API calls and kernel launches lack error checking, which can hide failures until they corrupt later steps.
- Output writes drop all files into the working directory without subfolders or metadata, making multi-run management cumbersome.

Addressing these items is recommended before large-scale studies.

---

## Troubleshooting & Profiling Tips
- **Runtime Failures:** Enable CUDA error checking after every API call (`cudaGetLastError`) to detect invalid allocations or launches early.
- **Performance:** Use Nsight Compute to confirm memory bandwidth usage of key kernels (`k_ibm_spread`, `k_collide`). Ensure coalesced memory access by aligning gel node counts with warp sizes where possible.
- **I/O Bottlenecks:** The asynchronous writer thread can lag behind on large grids. Consider batching outputs or writing in binary to reduce file size.
- **Portability:** Audit any future platform-specific additions (filesystem, threading, timing) and guard them appropriately so the solver builds cleanly on both Windows and Linux toolchains.

---

For further questions or contributions, please open an issue or submit a pull request describing your proposed enhancements.
