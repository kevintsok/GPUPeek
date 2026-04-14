#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// CUDA Graph Research Kernels
// =============================================================================
//
// CUDA Graph API (cudaRuntime.h):
// - cudaGraphCreate - Create graph
// - cudaGraphInstantiate - Create executable graph
// - cudaGraphLaunch - Launch graph
// - cudaGraphExecDestroy - Destroy executable graph
// - cudaGraphDestroy - Destroy graph
//
// Use cases:
// - Kernel launch overhead reduction
// - Batch processing optimization
// - Streaming workloads
// - Deep learning inference
//
// Benefits:
// - Eliminates CPU overhead for multiple kernel launches
// - Enables parallel kernel execution
// - Reduces latency for repeated workloads
// =============================================================================

// =============================================================================
// Simple Kernels for Graph Testing
// =============================================================================

template <typename T>
__global__ void vectorAddKernel(const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 T* __restrict__ c,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void vectorMulKernel(const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 T* __restrict__ c,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}

template <typename T>
__global__ void vectorScaleKernel(T* __restrict__ data,
                                   T scalar,
                                   size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = data[idx] * scalar;
    }
}

template <typename T>
__global__ void matrixMultiplyKernel(const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      T* __restrict__ C,
                                      size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <typename T>
__global__ void reluKernel(T* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = data[idx] > 0 ? data[idx] : 0;
    }
}

template <typename T>
__global__ void biasAddKernel(T* __restrict__ data,
                                 const T* __restrict__ bias,
                                 size_t N, size_t channels) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        size_t channel = idx % channels;
        data[idx] = data[idx] + bias[channel];
    }
}

// =============================================================================
// CUDA Graph Helper Functions
// =============================================================================

// Structure to hold CUDA Graph objects
struct GraphBenchmarkData {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStream_t stream;

    GraphBenchmarkData() : graph(nullptr), exec(nullptr), stream(nullptr) {}
};

// Create a simple graph with add + scale kernels
cudaError_t createSimpleGraph(cudaGraph_t* graph,
                               const float* h_a, const float* h_b,
                               float* h_c, size_t N) {
    cudaError_t err;

    // Create graph
    err = cudaGraphCreate(graph, 0);
    if (err != cudaSuccess) return err;

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    err = cudaMalloc(&d_a, N * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_b, N * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_c, N * sizeof(float));
    if (err != cudaSuccess) return err;

    // Copy data
    err = cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // For simplicity, we'll create the graph using launch patterns
    // In real code, you would use cudaGraphAddKernelNode, etc.

    // Store pointers for later use (in real code, these would be in graph nodes)
    // This is a simplified example - actual graph creation requires more setup

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return cudaSuccess;
}

// =============================================================================
// Graph Node Addition Functions
// =============================================================================

// Add a kernel node to a graph
cudaError_t addKernelNode(cudaGraph_t graph,
                          const void* func,
                          void** args,
                          size_t argCount,
                          dim3 gridDim,
                          dim3 blockDim,
                          size_t sharedMem = 0) {
    cudaError_t err;

    cudaKernelNodeParams params;
    memset(&params, 0, sizeof(params));

    params.func = func;
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.sharedSizeBytes = sharedMem;
    params.args = args;

    cudaGraphNode_t node;
    err = cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);

    return err;
}

// Add a memset node to a graph
cudaError_t addMemsetNode(cudaGraph_t graph,
                          void* devPtr,
                          int value,
                          size_t count) {
    cudaError_t err;

    cudaMemsetParams params;
    memset(&params, 0, sizeof(params));

    params.dst = devPtr;
    params.value = value;
    params.count = count;

    cudaGraphNode_t node;
    err = cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &params);

    return err;
}

// Add a memcpy node to a graph
cudaError_t addMemcpyNode(cudaGraph_t graph,
                         void* dst, const void* src,
                         size_t count,
                         cudaMemcpyKind kind) {
    cudaError_t err;

    cudaMemcpyParams params;
    memset(&params, 0, sizeof(params));

    params.dst = dst;
    params.src = src;
    params.count = count;
    params.kind = kind;

    cudaGraphNode_t node;
    err = cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &params);

    return err;
}

// =============================================================================
// Conditional Graph Nodes
// =============================================================================

// Create a graph with conditional execution
// Uses cudaGraphConditionalHandle for if-else patterns
cudaError_t createConditionalGraph(cudaGraph_t* graph,
                                   void* conditionPtr,
                                   size_t N) {
    cudaError_t err;

    // Create graph
    err = cudaGraphCreate(graph, 0);
    if (err != cudaSuccess) return err;

    // Create conditional handles for branches
    cudaGraphConditionalHandle handle;
    err = cudaGraphConditionalHandleCreate(&handle, *graph, 0, 0);
    if (err != cudaSuccess) return err;

    // Add conditional node
    // In real implementation, would use cudaGraphAddNode to add
    // conditional nodes with the handle

    return cudaSuccess;
}

// =============================================================================
// Graph Update Functions
// =============================================================================

// Update graph parameters (for dynamic graphs)
// cudaGraphExecUpdate allows updating parameters without re-instantiating
cudaError_t updateGraphNode(cudaGraphExec_t exec,
                             cudaGraphNode_t node,
                             const void* func,
                             void** args,
                             dim3 gridDim,
                             dim3 blockDim) {
    cudaError_t err;

    cudaKernelNodeParams params;
    memset(&params, 0, sizeof(params));

    params.func = func;
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.args = args;

    err = cudaGraphExecKernelNodeSetParams(exec, node, &params);

    return err;
}

// =============================================================================
// Stream Capture Functions
// =============================================================================

// Begin stream capture for graph creation
// cudaStreamBeginCapture marks a stream for graph creation
cudaError_t beginStreamCapture(cudaStream_t stream) {
    return cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
}

// End stream capture and create graph
// cudaStreamEndCapture completes capture and returns graph
cudaError_t endStreamCapture(cudaStream_t stream, cudaGraph_t* graph) {
    return cudaStreamEndCapture(stream, graph);
}

// Check if stream is being captured
// cudaStreamIsCapturing returns capture status
cudaError_t isStreamCapturing(cudaStream_t stream, int* isCapturing) {
    cudaStreamCaptureStatus status;
    cudaError_t err = cudaStreamGetCaptureInfo(stream, &status, nullptr);
    if (err != cudaSuccess) return err;
    *isCapturing = (status == cudaStreamCaptureStatusActive);
    return cudaSuccess;
}
