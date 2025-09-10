# pscan_torch
Learning GPU programming with Mojo on parallel scan.

This project demonstrates how to wrap Mojo kernels and expose them to PyTorch through custom operations. It implements parallel prefix sum (scan) algorithms as an educational example.

## Usage
Run tests: `pixi run test-all-wrappers` (also installs dependencies if needed)

## Features
- Single-block and multi-block parallel prefix sum implementations in Mojo
- PyTorch wrapper functions using MAX's CustomOpLibrary
- CUDA and ROCm support through Pixi environments
- Test suite comparing results against NumPy reference implementation

## Requirements
- Python 3.12
- Mojo
- CUDA 12.x or ROCm 6.3
- PyTorch 2.7.1

## Project Structure
- `op/`: Mojo kernel implementations
- `wrappers.py`: PyTorch wrapper functions for Mojo kernels
- `test_wrappers.py`: Test suite for kernel implementations