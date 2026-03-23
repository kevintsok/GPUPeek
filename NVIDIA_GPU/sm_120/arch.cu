#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// SM 12.0 (Blackwell) architecture-specific utilities

namespace sm_120 {

struct ArchInfo {
    const char* name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t l2_cache_size;
    size_t l3_cache_size;  // Not always available
    int max_threads_per_sm;
    int max_registers_per_sm;
    int num_tensor_cores_per_sm;
    int num_rt_cores;  // Ray tracing cores (if available)
};

__host__ ArchInfo getArchInfo() {
    ArchInfo info = {};
    info.name = "Blackwell (SM 12.0)";
    info.compute_capability_major = 12;
    info.compute_capability_minor = 0;
    info.max_threads_per_sm = 1536;
    info.max_registers_per_sm = 65536;
    info.num_tensor_cores_per_sm = 0;  // Not exposed via basic API
    info.num_rt_cores = 0;

    // Blackwell L2 cache varies by exact SKU
    // RTX 5080 Laptop typically has 5120 KB L2
    info.l2_cache_size = 5 * 1024 * 1024;  // 5 MB estimate

    return info;
}

__host__ void printArchSpecificInfo() {
    ArchInfo info = getArchInfo();
    printf("  Architecture:         %s\n", info.name);
    printf("  L2 Cache Size:       %zu MB\n", info.l2_cache_size / (1024 * 1024));
    printf("  Max Threads/SM:      %d\n", info.max_threads_per_sm);
    printf("  Max Registers/SM:   %d\n", info.max_registers_per_sm);
}

}  // namespace sm_120
