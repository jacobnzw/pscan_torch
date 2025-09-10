Notes and Learnings
===================

From Augment Chat (🧂)
----------------------
- Grid divides into thread blocks, blocks into warps, warps into threads.
  - Elements within grid are indexed by 3D coordinates (x, y, z)
- Threads have global and block-local indexes.
- Warps are the smallest unit of execution: threads that run in lockstep on the same SIMD unit.
- Thread divergence happens when different threads do different amounts of work (eg. loop divergence). Control flow divergence and data-dependent branching also cause divergence.
- Memory Hierarchy: Registers, Shared Memory, Global Memory
    - Registers are the fastest, but limited in quantity. 
    - Shared memory is slower but can be accessed by multiple threads within a block. 
    - Global memory is the slowest but largest.
- Arithmetic intensity: the ratio of floating-point operations to memory operations. 
    - High arithmetic intensity is desirable. Achieved by minimizing fetches from global memory.
    - How many ops per byte fetched from global memory? 
- Work-Efficiency: the number of ops performed by all threads shouldn't exceed the number of ops needed to solve the problem by a sequential algorithm.
- Shared memory:
    - Shared by threads within a block
    - Organized into banks, enabling access to multiple entries at once
    - \# banks = \# threads per warp
    - Bank conflicts occur when multiple threads access the same bank
    - Minimize bank conflicts for high shared memory bandwidth
- Good GPU kernel design involves making kernels size-agnostic by separating the algorithm logic from the execution configuration.
- Even with multi-block implementations, you'll eventually hit hardware limits:

    - Max grid dimensions: typically 65535 x 65535 x 65535 blocks
    - Max threads per block: 1024-2048 depending on GPU
    - Max total threads: ~2 billion on modern GPUs

    When you exceed these limits:

    - Chunked processing: Break input into multiple kernel launches
    - Hierarchical approach: Multiple levels of prefix sum
    - CPU-GPU hybrid: Process chunks on GPU, combine on CPU
- The "ceiling division trick" to round up to the next multiple of warp size:
    
    `((size + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE`
- Shared memory allocation requires compile-time constants because the GPU needs to know the memory layout at kernel compilation time.
- Grid dimensions (number of blocks) and threads per block can be computed at runtime and passed to the kernel launch configuration.
- What if the input size < warp size?
    - Thread underutilization: Only 10/32 threads do work
    - Warp divergence: Inactive threads still consume resources
    - Overhead: Launching 32 threads for 10 elements is wasteful
    - For very small inputs, the GPU launch overhead often exceeds the computation benefit anyway.


Resources
=========

## Videos

- [Udacity: Thread Blocks and GPU Hardware](https://youtu.be/usY0643pYs8?si=h_lmI4BHcCOpaCEk)
- [Intro to CUDA: High-Level Concepts](https://youtu.be/4APkMJdiudU?si=mvo900BjRYnuW5cb)
- [Udacity: GPU Memory Model](https://youtu.be/HQejUtJtBlg?si=KmtDZ29btA-SxPm4)
- [Intro to CUDA: Memory Model](https://youtu.be/OSpy-HoR0ac?si=ZKC6P6_oWSKL7q0c)

## Mojo
- [Mojo GPU Puzzle 9: Debugging](https://puzzles.modular.com/puzzle_09/puzzle_09.html)
- [Mojo GPU Puzzle 14: Prefix Sum](https://puzzles.modular.com/puzzle_14/puzzle_14.html)

## NVidia
- [GPU Gems 3: Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [CUDA UnBound](https://nvidia.github.io/cccl/cub/index.html)
- [Installing CUDA TOOLKIT on WSL Linux](https://youtu.be/JaHVsZa2jTc?si=ALfZD3eMfqSXCD-J)
