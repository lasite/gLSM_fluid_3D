# gLSM Fluid 3D Project Documentation

This document is a practical guide to the **gLSM Fluid 3D** CUDA code base. It summarizes how the solver couples an active gel model to a three-dimensional lattice Boltzmann (LBM) fluid, how to build and run the code, what defaults are hard-coded, and where to look when extending the solver.

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
The simulation resolves a deformable, chemically active gel immersed in a viscous fluid. The gel is modeled on a Lagrangian mesh whose nodes store geometry (`rn`, `rm`), chemistry (`um`, `vm`, `wm`), and mechanics (`Veln`, `Fn`). The surrounding fluid advances on a D3Q19 lattice. Gel–fluid coupling uses an immersed boundary method (IBM): gel boundary forces are spread to the fluid grid, fluid velocities are interpolated back to the gel nodes, and both subsystems iterate until the next macro time step.

---

## Codebase Layout
| File / Directory | Description |
| ---------------- | ----------- |
| `sim.cpp` | Program entry point. Instantiates two gels, a `Coupler`, and a `Fluid`, then iterates the coupling loop for `runstep = 10000` steps. |
| `gel.h` / `gel.cu` | Gel data model and update routines. Allocates host/device buffers for nodal geometry, forces, and chemistry; launches CUDA kernels declared in `gel_kernels.cuh`; and writes gel-specific outputs (`gel<id>rnXXXX.dat`, etc.). |
| `fluid.h` / `fluid.cu` | LBM-based fluid simulator. Owns distribution functions, macroscopic fields, and immersed boundary force buffers; advances velocity and concentration; and writes `VelbXXXX.dat` / `ConcXXXX.dat`. |
| `coupling.h` / `coupling.cu` | IBM glue between gels and fluid. Packs gel boundary data, spreads forces to the fluid grid, samples velocities back, and holds shared buffers for both subsystems. |
| `*_kernels.cu` / `*_kernels.cuh` | CUDA kernels and declarations for gel mechanics/chemistry, fluid collide/stream, and coupling interpolation/spreading operations. |
| `gLSM_fluid_3D.sln` / `gLSM_fluid_3D.vcxproj` | Visual Studio + CUDA project files for Windows builds. |

The repository currently ships only source and project files; tests and post-processing scripts are not included.

---

## Build & Toolchain Guidance
### Windows (Visual Studio)
1. Install the NVIDIA CUDA Toolkit (10.2 or higher) alongside Visual Studio 2019 or newer.
2. Open `gLSM_fluid_3D.sln`, ensure the CUDA build customization is enabled, and select a GPU architecture (e.g., `sm_70` or newer).
3. Build the `Release` configuration. Output binaries land next to the solution file.

### Linux (command-line)
No CMake file is provided. Build with `nvcc`, compiling the entry point, subsystem sources, and their kernels:
```bash
nvcc -O3 -std=c++17 -arch=sm_70 \
     sim.cpp gel.cu fluid.cu coupling.cu \
     gel_kernels.cu fluid_kernels.cu coupling_kernels.cu \
     -o gLSM_fluid_3D
```
If the executable crashes with `cudaSetDevice(1)` errors, change the hard-coded device in `sim.cpp` to `0` (or any available device ID). Ensure `CUDA_HOME` is set and the host compiler supports C++17.

### Common Requirements
- CUDA-capable GPU with enough memory for a `101×101×101` lattice plus two 6×6×6 gels (hundreds of MB).
- Adequate CPU RAM for pinned host buffers and asynchronous file output threads.
- Optional: Nsight Compute / Nsight Systems for profiling, and Python/Matlab for post-processing `.dat` outputs.

---

## Execution Pipeline
The top-level loop in `sim.cpp` follows this sequence:

1. **Initialization**
   - Two `Gel` instances are built with fixed sizes (`6×6×6` elements), positions (`(15,15,15)` and `(35,35,35)`), and types (`NIPAAM` and `PAAM`).
   - A `Coupler` aggregates their boundary data and exposes buffers to the fluid.
   - A `Fluid` initializes a `101×101×101` lattice and associated IBM force/velocity buffers.

2. **Main Time Step (`for (int iter = 0; iter <= runstep; ++iter)`)**
   - *Gel updates:* `stepElasticity(iter)` then `stepChemistry(iter)` advance mechanical and reaction–diffusion fields on each gel.
   - *Coupling pack:* `Coupler::packFromGels()` collects boundary positions/forces into contiguous device buffers.
   - *Fluid step:* `Fluid::stepVelocity(iter)` and `Fluid::stepConcentration()` perform LBM collide/stream and scalar advection/diffusion with IBM forcing.
   - *Coupling scatter:* `Coupler::scatterToGels()` samples fluid velocities back to the gel boundaries.
   - *Diagnostics:* Every 1000 iterations, gels and fluid copy data to the host and spawn writer threads to dump `.dat` snapshots.

3. **Finalization**
   - After `runstep = 10000`, host/device resources are freed and any outstanding writer threads are joined.

---

## Runtime Configuration
All runtime parameters are currently hard-coded:

- **Grid Dimensions:** `fluidSize = (101,101,101)` and two `gelSize = (6,6,6)` meshes positioned at `(15,15,15)` and `(35,35,35)`.
- **Time Control:** `runstep = 10000` iterations. Both gels and fluid step with `dt = 1e-3` (stored as `m_dt` for gels and `dt` for the fluid). The fluid computes `Nsub` immersed boundary substeps from its viscosity.
- **Material Parameters:** `GelParams` and `FluidParams` are allocated and populated with constants inside their constructors; no file or CLI overrides exist.
- **Device Selection:** `cudaSetDevice(1)` assumes a second GPU. Change to `0` if only one device is present.

Externalizing these settings (e.g., JSON/INI or command-line arguments) would make the solver easier to reproduce and configure.

---

## Key Data Structures
### Gel-related
- `m_hum`, `m_hvm`, `m_hwm` (host) and `m_dum`, `m_dvm`, `m_dwm` (device) store per-element chemistry fields.
- `m_hrn`, `m_hVeln`, `m_hFn` and their device counterparts hold nodal geometry, velocity, and force vectors; `m_hrm`/`m_drm` capture element centers.
- `m_hmap_node`, `m_hmap_element`, and `m_hbIndex` (plus GPU mirrors) encode boundary node/element indices and neighbor lookups for IBM interactions.
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
- `d_A`: Intended auxiliary array for IBM spreading (declared but not allocated—see [Implementation Caveats](#implementation-caveats)).

---

## Output Artefacts
Outputs are written to the working directory every 1000 iterations. Filenames encode the gel ID and an integer-valued time stamp computed as `int(iter * dt)`:

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
2. **Portable Runtime:** Expose GPU selection, run length, and output locations via arguments or configuration files instead of relying on in-source constants such as `cudaSetDevice(1)` and `runstep`.
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

---

## Troubleshooting & Profiling Tips
- **Runtime Failures:** Enable CUDA error checking after every API call (`cudaGetLastError`) to detect invalid allocations or launches early.
- **Performance:** Use Nsight Compute to confirm memory bandwidth usage of key kernels (`k_ibm_spread`, `k_collide`). Ensure coalesced memory access by aligning gel node counts with warp sizes where possible.
- **I/O Bottlenecks:** The asynchronous writer thread can lag behind on large grids. Consider batching outputs or writing in binary to reduce file size.
- **Portability:** Audit any future platform-specific additions (filesystem, threading, timing) and guard them appropriately so the solver builds cleanly on both Windows and Linux toolchains.

---

For further questions or contributions, please open an issue or submit a pull request describing your proposed enhancements.
