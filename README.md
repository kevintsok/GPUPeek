# GPUPeek

A CUDA benchmark framework for exploring GPU mechanisms and metrics.

## Project Structure

```
GPUPeek/
├── CMakeLists.txt              # Global build configuration
├── README.md
├── CLAUDE.md                     # Project rules and conventions
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
    ├── metal/                    # Apple Metal GPU (M1/M2/M3/M4 series)
    │   ├── RESEARCH.md           # Research documentation
    │   ├── bandwidth_test.metal  # Memory bandwidth kernels
    │   ├── bandwidth_host.mm     # Host code for bandwidth tests
    │   └── compute_test.metal    # Compute throughput kernels
    └── sm_120/                  # SM 12.0 (Blackwell) specific
        ├── arch.cu               # Architecture info & utilities
        ├── arch_kernels.cu       # Architecture-specific kernels
        ├── benchmarks.cu          # Architecture-specific benchmark runner
        ├── memory_research_kernel.cu    # Memory research kernels
        ├── memory_research_benchmarks.cu
        ├── deep_research_kernel.cu      # Deep research kernels
        ├── deep_research_benchmarks.cu
        ├── advanced_research_kernel.cu   # Advanced research kernels
        ├── advanced_research_benchmarks.cu
        ├── ncu_profiling_kernel.cu      # NCU profiling kernels
        ├── ncu_profiling_benchmarks.cu
        ├── cuda_core_kernels.cu         # CUDA Core arithmetic kernels
        ├── cuda_core_benchmarks.cu
        ├── atomic_kernels.cu            # Atomic research kernels
        ├── atomic_benchmarks.cu
        ├── barrier_kernels.cu           # Barrier sync kernels
        ├── barrier_benchmarks.cu
        ├── warp_specialize_kernels.cu   # Warp specialization kernels
        ├── warp_specialize_benchmarks.cu
        ├── mma_research_kernel.cu      # MMA research kernels
        ├── mma_research_benchmarks.cu
        ├── tensor_mem_research_kernel.cu   # Tensor memory kernels
        ├── tensor_mem_research_benchmarks.cu
        ├── dp4a_research_kernel.cu        # DP4A (INT8 dot) kernels
        ├── dp4a_research_benchmarks.cu
        ├── wgmma_research_kernel.cu       # WGMMA (Async warpgroup MMA) kernels
        ├── wgmma_research_benchmarks.cu
        ├── fp8_research_kernel.cu        # FP8 / TCGen05 Block Scaling kernels
        ├── fp8_research_benchmarks.cu
        ├── cuda_graph_research_kernel.cu   # CUDA Graph kernels
        ├── cuda_graph_research_benchmarks.cu
        ├── unified_memory_research_kernel.cu   # Unified Memory kernels
        ├── unified_memory_research_benchmarks.cu
        ├── multi_stream_research_kernel.cu   # Multi-Stream kernels
        ├── multi_stream_research_benchmarks.cu
        ├── mbarrier_research_kernel.cu   # Mbarrier (memory barrier) kernels
        ├── mbarrier_research_benchmarks.cu
        ├── cooperative_groups_research_kernel.cu   # Cooperative Groups kernels
        ├── cooperative_groups_research_benchmarks.cu
        ├── redux_sync_research_kernel.cu   # Redux.sync warp reduction kernels
        ├── redux_sync_research_benchmarks.cu
        ├── fp4_fp6_research_kernel.cu   # FP4/FP6 low-precision MMA kernels
        └── fp4_fp6_research_benchmarks.cu
```

## GPU Architecture Support

Each GPU architecture (compute capability) has its own directory:
- `metal/` - Apple Metal GPU (M1/M2/M3/M4 series)
- `sm_120/` - Blackwell (RTX 5080, RTX 5070, etc.)
- `sm_90/` - Ada Lovelace (RTX 4090, RTX 4080, etc.)
- `sm_80/` - Ampere (RTX 3090, A100, etc.)
- `sm_70/` - Volta/Vega (V100, etc.)
- ... (can be extended as needed)

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

# Build wmma module (Tensor Core benchmark)
cd ../wmma && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./wmma_final_benchmark  # Tensor Core performance benchmark
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
./wmma_final_benchmark    # Tensor Core benchmark (WMMA)

# Example: Run memory research with 1M elements
./gpupeek_memory 1048576

# Example: Run Tensor Core benchmark
./wmma_final_benchmark
```

## NCU Profiling

```bash
# Profile Tensor Core utilization
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./wmma_final_benchmark

# Memory bandwidth analysis
ncu --set full --metrics dram__bytes.sum ./gpupeek_memory
```

## Target GPU

- **GPU**: NVIDIA GeForce RTX 5080 Laptop GPU
- **Architecture**: Blackwell (Compute Capability 12.0)
- **CUDA**: 13.0
- **Driver**: 595.79
