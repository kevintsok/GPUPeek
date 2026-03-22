# GPUPeek

A CUDA benchmark framework for exploring GPU mechanisms and metrics.
Uses compute capability-specific directories for architecture-specific code.

## Project Structure

```
GPUPeek/
├── CMakeLists.txt
├── README.md
├── ref/                         # NVIDIA official documentation
│   └── README.md                 # Documentation index
└── src/
    ├── common/                   # Architecture-agnostic code
    │   ├── main.cu              # Main program (auto-detects GPU)
    │   ├── gpu_info.h/cu        # GPU information utilities
    │   └── timer.h              # GPU/CPU timing utilities
    ├── generic/                  # Generic kernels (work on all GPUs)
    │   ├── bandwidth_kernel.cu   # Memory bandwidth kernels
    │   ├── compute_kernel.cu     # Compute throughput kernels
    │   └── warp_kernel.cu       # Warp-level operation kernels
    └── sm_120/                  # SM 12.0 (Blackwell) specific
        ├── arch.cu               # Architecture info & utilities
        ├── arch_kernels.cu       # Architecture-specific kernels
        └── benchmarks.cu          # Architecture-specific benchmark runner
```

## Architecture-Specific Directories

Each GPU architecture (compute capability) has its own directory:
- `sm_120/` - Blackwell (RTX 5080, RTX 5070, etc.)
- `sm_90/` - Ada Lovelace (RTX 4090, RTX 4080, etc.)
- `sm_80/` - Ampere (RTX 3090, A100, etc.)
- `sm_70/` - Volta/Vega (V100, etc.)
- ... (can be extended as needed)

## Building

### Prerequisites

- NVIDIA CUDA Toolkit 13.0+
- Visual Studio 2022 with C++ toolchain (Windows)
- NVIDIA Driver 535+

### Compile (Windows with Visual Studio)

```bash
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/bin/HostX64/x64:$PATH"

"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" \
    -o build/gpupeek \
    src/common/main.cu src/common/gpu_info.cu \
    -Isrc/common -Iinclude \
    -O3 -arch=sm_90 --use_fast_math
```

### CMake (alternative)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
```

## Running

```bash
# Run all benchmarks (generic + architecture-specific)
./build/gpupeek

# Run only generic benchmarks (work on all architectures)
./build/gpupeek generic

# Run only architecture-specific benchmarks (auto-detects GPU)
./build/gpupeek arch

# Run with custom size
./build/gpupeek all 10000000
```

## Benchmarks

### Generic (All Architectures)

**Memory Bandwidth**
- Sequential Read
- Sequential Write
- Read-Modify-Write

**Compute Throughput**
- FP32 FMA (fused multiply-add)
- INT32 Arithmetic

**Warp-Level Operations**
- Warp Shuffle
- Warp Reduction
- Warp Vote

### SM 12.0 (Blackwell) Specific

**Enhanced Warp Operations**
- Enhanced Shuffle

**Memory Operations**
- Async Copy
- L2 Streaming
- Register Bandwidth
- Software Prefetch
- Reduced Precision

## Sample Output

```
=== GPU Info (Device 0) ===
  Name:                  NVIDIA GeForce RTX 5080 Laptop GPU
  Compute Capability:    12.0
  Number of SMs:         60
  Cores per SM:          128
  Total Cores:           7680
  Global Memory:         15.92 GB
  Shared Mem per Block:   48 KB
  Max Threads per Block: 1024
  Max Threads per SM:    1536
  Registers per Block:   65536
  Warp Size:             32
===========================

Detected Compute Capability: SM 12.0 (0x78)

=== Memory Bandwidth Benchmarks (Generic) ===
Config: 4096 blocks x 256 threads = 1048576 threads, 4.00 MB
Sequential Read:    3.84 GB/s (1.091 ms per kernel)
Sequential Write:   366.53 GB/s (0.011 ms per kernel)
Read-Modify-Write:  402.18 GB/s (0.010 ms per kernel)

=== Compute Throughput Benchmarks (Generic) ===
FP32 FMA:           88.55 GFLOPS (0.012 ms per kernel)
INT32 Arithmetic:   106.38 GIOPS (0.010 ms per kernel)

=== Warp-Level Benchmarks (Generic) ===
Warp Shuffle:       305.59 GB/s (0.014 ms per kernel)
Warp Reduction:     0.015 ms per kernel
Warp Vote:          0.015 ms per kernel

[Using SM 12.0 (Blackwell) specific benchmarks]

=== SM 12.0 (Blackwell) Specific Benchmarks ===
  Architecture:         Blackwell (SM 12.0)
  L2 Cache Size:       5 MB
  Max Threads/SM:      1536
  Max Registers/SM:   65536

--- Enhanced Warp Operations ---
Enhanced Shuffle:    418.49 GB/s (0.010 ms)

--- Memory Operations ---
Async Copy:          422.69 GB/s (0.010 ms)
L2 Streaming:        316.46 GB/s (0.013 ms)
Register Bandwidth:  298.96 GB/s (0.014 ms)
Software Prefetch:   251.10 GB/s (0.017 ms)
Reduced Precision:   303.11 GB/s (0.014 ms)
```

## Adding Support for a New Architecture

1. Create a new directory: `src/sm_XX/` where XX is the compute capability * 10
   e.g., `src/sm_90/` for SM 9.0 (Ada Lovelace)

2. Create architecture-specific files:
   - `arch.cu` - Architecture info and utilities
   - `arch_kernels.cu` - Architecture-specific kernel implementations
   - `benchmarks.cu` - Architecture-specific benchmark runner

3. Update `src/common/main.cu`:
   - Add include for the new architecture's benchmarks
   - Add case in the switch statement for the new architecture

## GPU Target

- **GPU**: NVIDIA GeForce RTX 5080 Laptop GPU
- **Architecture**: Blackwell (Compute Capability 12.0)
- **CUDA**: 13.0
- **Driver**: 595.79
