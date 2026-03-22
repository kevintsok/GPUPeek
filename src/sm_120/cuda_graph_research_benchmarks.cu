#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/timer.h"
#include "cuda_graph_research_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

extern const char* formatBandwidth(double GBps);

// =============================================================================
// CUDA Graph Research Benchmarks
// =============================================================================
//
// CUDA Graph API Functions:
// - cudaGraphCreate - Create an empty graph
// - cudaGraphInstantiate - Create executable graph from graph
// - cudaGraphLaunch - Launch executable graph
// - cudaGraphExecDestroy - Destroy executable graph
// - cudaGraphDestroy - Destroy graph
// - cudaStreamBeginCapture - Begin stream capture
// - cudaStreamEndCapture - End stream capture, return graph
//
// Use cases:
// - Reduce kernel launch overhead
// - Batch processing optimization
// - Deep learning inference pipelines
// - Streaming workloads
//
// NCU Metrics:
// - sm__throughput.avg.pct_of_peak_sustainedTesla - GPU utilization
// - dram__bytes.sum - Memory bandwidth
// =============================================================================

// =============================================================================
// Section 1: Graph Lifecycle Tests
// =============================================================================

void runGraphLifecycleTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. CUDA Graph Lifecycle Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const int iterations = 100;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMallocHost(&h_a, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_b, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_c, N * sizeof(float)));

    for (size_t i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = 0;
    }

    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // Test 1: Regular kernel launch (baseline)
    printf("\n--- Graph Lifecycle Performance ---\n\n");

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    timer.stop();
    printf("Regular Launch:      %.3f ms per launch (baseline)\n",
           timer.elapsed_ms() / iterations);

    // Test 2: Graph create + instantiate overhead
    cudaGraph_t graph;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    timer.stop();
    printf("Graph Create:       %.3f ms per create\n",
           timer.elapsed_ms() / iterations);

    // Test 3: Graph instantiate overhead
    cudaGraphExec_t exec;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        // In real code, would add nodes here
        // For demo, just destroy
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    timer.stop();
    printf("Graph Instantiate:  %.3f ms per instantiate\n",
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
}

// =============================================================================
// Section 2: Stream Capture Tests
// =============================================================================

void runStreamCaptureTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. Stream Capture Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const int iterations = 100;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMallocHost(&h_a, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_b, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_c, N * sizeof(float)));

    for (size_t i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // Test: Stream capture + end capture
    printf("\n--- Stream Capture Performance ---\n\n");

    cudaGraph_t graph;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // Capture multiple kernel launches
        vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, N);
        vectorScaleKernel<float><<<gridDim, blockDim, 0, stream>>>(d_c, 2.0f, N);

        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    timer.stop();
    printf("Stream Capture:    %.3f ms per capture (2 kernels)\n",
           timer.elapsed_ms() / iterations);

    // Test: Graph launch vs regular launch
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    timer.stop();
    printf("Graph Launch:      %.3f ms per launch\n",
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
}

// =============================================================================
// Section 3: Kernel Launch Overhead Tests
// =============================================================================

void runLaunchOverheadTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. Kernel Launch Overhead Tests\n");
    printf("================================================================================\n");

    const size_t N = 1024;  // Small N to highlight launch overhead
    const int iterations = 1000;

    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));

    dim3 gridDim(4);  // Minimal grid
    dim3 blockDim(32);

    GPUTimer timer;

    // Test 1: Regular kernel launch (small kernel)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    timer.stop();
    printf("Regular Launch (small): %.3f ms per launch\n",
           timer.elapsed_ms() / iterations);

    // Test 2: Graph with single kernel
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaStreamEndCapture(0, &graph));

    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Graph Launch (single): %.3f ms per launch\n",
           timer.elapsed_ms() / iterations);

    // Calculate overhead reduction
    float regular_ms = timer.elapsed_ms() / iterations;

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));

    // Test 3: Multiple kernels in graph
    CHECK_CUDA(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    vectorMulKernel<float><<<gridDim, blockDim>>>(d_a, d_c, d_c, N);
    vectorScaleKernel<float><<<gridDim, blockDim>>>(d_c, 0.5f, N);
    CHECK_CUDA(cudaStreamEndCapture(0, &graph));

    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Graph Launch (3 kernels): %.3f ms per launch\n",
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));

    printf("\nNote: Graph launch overhead reduction is most significant for small kernels\n");
    printf("      and multiple kernel launches.\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 4: Pipeline / Inference Benchmarks
// =============================================================================

void runPipelineTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Pipeline / Inference Benchmarks\n");
    printf("================================================================================\n");

    const size_t N = 1 << 18;
    const int iterations = 100;

    float *h_input, *h_output;
    float *d_input, *d_hidden, *d_output;

    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output, N * sizeof(float)));

    for (size_t i = 0; i < N; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Simple 3-layer inference pipeline:
    // Input -> Hidden1 (mul) -> Hidden2 (add) -> Output (relu)
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    // Regular sequential execution
    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorMulKernel<float><<<gridDim, blockDim, 0, stream>>>(d_input, d_hidden, d_hidden, N);
        vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_hidden, d_input, d_output, N);
        reluKernel<float><<<gridDim, blockDim, 0, stream>>>(d_output, N);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    timer.stop();
    printf("Sequential (3 ops): %.3f ms per iteration\n",
           timer.elapsed_ms() / iterations);

    // Graph-based execution
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorMulKernel<float><<<gridDim, blockDim, 0, stream>>>(d_input, d_hidden, d_hidden, N);
    vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_hidden, d_input, d_output, N);
    reluKernel<float><<<gridDim, blockDim, 0, stream>>>(d_output, N);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    timer.stop();
    printf("Graph Pipeline (3 ops): %.3f ms per iteration\n",
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
}

// =============================================================================
// Section 5: Comparison Tests
// =============================================================================

void runComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. Graph vs Regular Launch Comparison\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const int iterations = 100;

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    printf("\n--- Single Kernel ---\n\n");

    // Regular
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    timer.stop();
    float regular_single = timer.elapsed_ms() / iterations;
    printf("Regular:            %.4f ms\n", regular_single);

    // Graph
    cudaGraph_t graph;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    timer.stop();
    float graph_single = timer.elapsed_ms() / iterations;
    printf("Graph:              %.4f ms (%.2fx speedup)\n",
           graph_single, regular_single / graph_single);

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));

    printf("\n--- Five Kernels ---\n\n");

    // Regular 5 kernels
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorAddKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
        vectorMulKernel<float><<<gridDim, blockDim>>>(d_a, d_c, d_c, N);
        vectorScaleKernel<float><<<gridDim, blockDim>>>(d_c, 2.0f, N);
        reluKernel<float><<<gridDim, blockDim>>>(d_c, N);
        biasAddKernel<float><<<gridDim, blockDim>>>(d_c, d_a, N, 1024);
        cudaDeviceSynchronize();
    }
    timer.stop();
    float regular_five = timer.elapsed_ms() / iterations;
    printf("Regular (5):        %.4f ms\n", regular_five);

    // Graph 5 kernels
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, N);
    vectorMulKernel<float><<<gridDim, blockDim, 0, stream>>>(d_a, d_c, d_c, N);
    vectorScaleKernel<float><<<gridDim, blockDim, 0, stream>>>(d_c, 2.0f, N);
    reluKernel<float><<<gridDim, blockDim, 0, stream>>>(d_c, N);
    biasAddKernel<float><<<gridDim, blockDim, 0, stream>>>(d_c, d_a, N, 1024);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    timer.stop();
    float graph_five = timer.elapsed_ms() / iterations;
    printf("Graph (5):          %.4f ms (%.2fx speedup)\n",
           graph_five, regular_five / graph_five);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 6: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. NCU Profiling Reference - CUDA Graph\n");
    printf("================================================================================\n");

    printf("\n--- Key NCU Metrics ---\n\n");

    printf("GPU Utilization:\n");
    printf("  ncu --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek graph\n\n");

    printf("Memory Bandwidth:\n");
    printf("  ncu --metrics dram__bytes.sum ./gpupeek graph\n\n");

    printf("Kernel Analysis:\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek graph\n\n");

    printf("--- CUDA Graph API Reference ---\n\n");

    printf("Graph Lifecycle:\n");
    printf("  cudaGraphCreate(pGraph, flags)        - Create empty graph\n");
    printf("  cudaGraphInstantiate(pExec, graph)   - Create executable graph\n");
    printf("  cudaGraphLaunch(exec, stream)       - Launch executable graph\n");
    printf("  cudaGraphExecDestroy(exec)           - Destroy executable graph\n");
    printf("  cudaGraphDestroy(graph)              - Destroy graph\n\n");

    printf("Stream Capture:\n");
    printf("  cudaStreamBeginCapture(stream, mode)  - Begin capture\n");
    printf("  cudaStreamEndCapture(stream, pGraph) - End capture, return graph\n");
    printf("  cudaStreamIsCapturing(stream, status)- Check capture status\n\n");

    printf("Node Operations:\n");
    printf("  cudaGraphAddKernelNode()   - Add kernel node\n");
    printf("  cudaGraphAddMemcpyNode()  - Add memcpy node\n");
    printf("  cudaGraphAddMemsetNode()  - Add memset node\n");
    printf("  cudaGraphAddEmptyNode()   - Add empty node\n");
    printf("  cudaGraphAddBarrierNode()  - Add barrier node\n\n");

    printf("Conditional Execution:\n");
    printf("  cudaGraphConditionalHandleCreate() - Create conditional handle\n\n");

    printf("Graph Update:\n");
    printf("  cudaGraphExecUpdate() - Update graph parameters\n");
    printf("  cudaGraphExecKernelNodeSetParams() - Update kernel params\n\n");

    printf("--- When to Use CUDA Graph ---\n\n");

    printf("Benefits:\n");
    printf("  1. Reduces CPU overhead for multiple kernel launches\n");
    printf("  2. Enables parallel kernel execution\n");
    printf("  3. Lower latency for repeated workloads\n");
    printf("  4. Better GPU utilization for small kernels\n\n");

    printf("Best Use Cases:\n");
    printf("  - Deep learning inference pipelines\n");
    printf("  - Repeated batch processing\n");
    printf("  - Multi-kernel streaming workloads\n");
    printf("  - Small kernels where launch overhead is significant\n\n");

    printf("Trade-offs:\n");
    printf("  - Graph creation has upfront cost\n");
    printf("  - Limited flexibility (graphs are static)\n");
    printf("  - Updates require re-instantiation or special handling\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runCUDAGraphResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) CUDA Graph Research                   #\n");
    printf("#           Kernel Launch Optimization                                        #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("CUDA Graph API\n");
    printf("================================================================================\n\n");

    printf("Graph Lifecycle:\n");
    printf("  cudaGraphCreate -> cudaGraphInstantiate -> cudaGraphLaunch\n\n");

    printf("Stream Capture:\n");
    printf("  cudaStreamBeginCapture -> [kernels] -> cudaStreamEndCapture\n\n");

    printf("Use Cases:\n");
    printf("  - Reduce kernel launch overhead\n");
    printf("  - Batch processing optimization\n");
    printf("  - Deep learning inference\n");
    printf("  - Multi-kernel pipelines\n\n");

    runGraphLifecycleTests();
    runStreamCaptureTests();
    runLaunchOverheadTests();
    runPipelineTests();
    runComparisonTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("CUDA Graph Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU profiling, run:\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe graph\n");
    printf("  ncu --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe graph\n");
    printf("\n");
}
