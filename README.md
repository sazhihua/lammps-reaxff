## Redesign ReaxFF on ARM: Enabling High-Performance Billion-Atom Reactive Molecular Dynamics

### 1. Build Instructions

Run the following commands to configure the build environment using CMake. Ensure that you are targeting the ARMv9 architecture with SVE2 support.

```bash
mkdir build && cd build
cmake -D PKG_REAXFF=ON \
      -D PKG_SVE=ON \
      -D CMAKE_CXX_FLAGS="-O3 -funroll-loops -ffast-math -std=gnu++17 -march=armv9+sve2 -msve-vector-bits=512" \
      -D CMAKE_CXX_STANDARD=17 ../cmake
```

### 2. Compilation

Compile the source code to generate the `lmp` executable. Use the `-j` flag to accelerate the process through parallel compilation.

```bash
make -j
```

### 3. Running Benchmarks

Test cases are located in `src/SVE/TEST`. To utilize the optimized ARM version, include the `-sf sve` flag in your execution command.

**Memory and Solver Configuration:**

* **Potential & Array Scaling (`pair_style`):**
The optional keywords `safezone`, `mincap`, and `minhbonds` are used to scale internal ReaxFF arrays to prevent "segmentation faults". **It is recommended to set `safezone` to 1.4:**
`pair_style reaxff <control_file> safezone 1.4`
*(Note: The first parameter `<control_file>` can be a path or `NULL` depending on your potential setup.)*
* **QEq Solver (`cg_algo`):**
In the `fix qeq/reaxff` command (e.g., `fix 2 all qeq/reaxff ... cg_algo 1`), you can specify the CG implementation:
* `0`: Optimized CG implementation, recommended for single-thread execution.
* `1`: Optimized CG implementation with OpenMP support, recommended for multi-thread execution.
* `2`: Pipelined CG with OpenMP (asynchronous overlapping of communication and computation).
* `3`: Pipelined CG with OpenMP and **Weight-Based Dynamic Re-balancing**. Thread 0 manages MPI communication and dynamically assists in SpMV computation based on workload weights to eliminate idle time.

**Execution Examples:**

* **Pure MPI (64 MPI Processes):**
```bash
    OMP_SET_AFFINITY=1 OMP_PROC_BIND=true OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=1 \
    mpirun -np 64 -rankfile ../rankfile64x1 ../../../../build/lmp -in in.reaxff -sf sve
```


* **Hybrid MPI + OpenMP (8 MPI Processes x 8 OpenMP Threads):**

```bash
    OMP_SET_AFFINITY=1 OMP_PROC_BIND=true OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=8 \
    mpirun -np 8 -rankfile ../rankfile8x8 ../../../../build/lmp -in in.reaxff -sf sve
```

### 4. Output and Performance Metrics

The simulation results and performance data will be displayed in the terminal and simultaneously saved to `log.lammps` in the execution directory.

**Example Output:**
`Performance: 0.497 ns/day, 48.272 hours/ns, 92.070 timesteps/s, 35.355 katom-step/s`

> **Note:** The compilation flags, MPI/OpenMP configurations, and paths provided above are for demonstration purposes only. Please adjust them according to your specific hardware environment and research requirements. For more details on parameter settings and command usage, please refer to the [LAMMPS ReaxFF Documentation](https://docs.lammps.org/pair_reaxff.html).
