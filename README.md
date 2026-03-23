# GPUPeek

A CUDA benchmark framework for exploring GPU mechanisms and metrics.

## Project Structure

```
GPUPeek/
├── CMakeLists.txt              # Global build configuration
├── README.md
├── CLAUDE.md                  # Project rules and conventions
├── docs/                      # Research reports
├── include/                   # Header files
├── NVIDIA_GPU/               # NVIDIA GPU code
│   ├── ref/                  # NVIDIA official documentation
│   ├── common/               # Architecture-agnostic code
│   │   ├── main.cu          # Main program (auto-detects GPU)
│   │   ├── gpu_info.h/cu   # GPU information utilities
│   │   └── timer.h         # GPU/CPU timing utilities
│   ├── generic/              # Generic kernels (work on all GPUs)
│   │   ├── bandwidth_kernel.cu
│   │   ├── compute_kernel.cu
│   │   └── warp_kernel.cu
│   └── sm_120/              # SM 12.0 (Blackwell)
│       ├── arch.cu          # Architecture info
│       ├── arch_kernels.cu  # Architecture-specific kernels
│       ├── benchmarks.cu    # Benchmark runner
│       └── [modules]/        # Independent research modules
│           ├── CMakeLists.txt # Each module can build independently
│           ├── main.cu       # Module entry point
│           ├── README.md     # Module documentation
│           ├── RESEARCH.md    # Research findings
│           └── *_kernel.cu   # Module source code
└── APPLE_GPU/               # Apple GPU support (future)
```

## GPU Architecture Support

- `NVIDIA_GPU/sm_120/` - Blackwell (RTX 5080, RTX 5070, etc.)
- `NVIDIA_GPU/sm_90/` - Ada Lovelace (RTX 4090, RTX 4080, etc.)
- `NVIDIA_GPU/sm_80/` - Ampere (RTX 3090, A100, etc.)
- `NVIDIA_GPU/sm_70/` - Volta/Vega (V100, etc.)

## Building

### Build Individual Modules (Recommended)

Each research module can be built independently:

```bash
# Build memory module
cd NVIDIA_GPU/sm_120/memory
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_memory

# Build wmma module
cd ../wmma && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_wmma
```

### Available Modules

| Module | Path | Description |
|--------|------|-------------|
| memory | `sm_120/memory/` | Memory subsystem research |
| wmma | `sm_120/wmma/` | WMMA/Tensor Core research |
| cuda_core | `sm_120/cuda_core/` | CUDA Core compute research |
| atomic | `sm_120/atomic/` | Atomic operations research |
| barrier | `sm_120/barrier/` | Barrier synchronization research |
| warp_specialize | `sm_120/warp_specialize/` | Warp specialization |
| tensor_mem | `sm_120/tensor_mem/` | Tensor memory operations |
| wgmma | `sm_120/wgmma/` | WGMMA (Hopper only) |
| dp4a | `sm_120/dp4a/` | DP4A research |
| fp8 | `sm_120/fp8/` | FP8 research |
| fp4_fp6 | `sm_120/fp4_fp6/` | FP4/FP6 research |
| deep | `sm_120/deep/` | Deep research (L2, TMA) |
| advanced | `sm_120/advanced/` | Advanced research |
| cooperative_groups | `sm_120/cooperative_groups/` | Cooperative Groups |
| mbarrier | `sm_120/mbarrier/` | MBarrier research |
| redux_sync | `sm_120/redux_sync/` | Redux.sync research |
| cuda_graph | `sm_120/cuda_graph/` | CUDA Graph research |
| unified_memory | `sm_120/unified_memory/` | Unified Memory research |
| multi_stream | `sm_120/multi_stream/` | Multi-Stream concurrency |
| ncu_profiling | `sm_120/ncu_profiling/` | NCU profiling research |

### CMake Build (Optional)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
```

## Running Benchmarks

```bash
# Run a specific module
./gpupeek_memory [elements]
./gpupeek_wmma [elements]

# Example: Run memory research with 1M elements
./gpupeek_memory 1048576
```

## NCU Profiling

```bash
# Profile a module
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek_wmma

# Memory bandwidth analysis
ncu --set full --metrics dram__bytes.sum ./gpupeek_memory
```

## Target GPU

- **GPU**: NVIDIA GeForce RTX 5080 Laptop GPU
- **Architecture**: Blackwell (Compute Capability 12.0)
- **CUDA**: 13.0
- **Driver**: 595.79
