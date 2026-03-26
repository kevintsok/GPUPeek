import Foundation
import Metal

// MARK: - Shader Compilation and Kernel Launch Benchmark

let shaderCompilerShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void simple_add(device float* out [[buffer(0)]],
                device const float* in [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = in[id] + 1.0f;
}
"""

public struct ShaderCompilerBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Shader Compilation and Kernel Launch Overhead")
        print("Compilation time vs execution time analysis")
        print(String(repeating: "=", count: 70))

        let iterations = 100
        let workSize = 65536

        print("\n--- Library Compilation Overhead ---")

        let compileStart = getTimeNanos()
        guard let compiledLibrary = try? device.makeLibrary(source: shaderCompilerShaders, options: nil) else {
            print("Failed to compile library")
            return
        }
        let compileEnd = getTimeNanos()
        let compileTime = getElapsedSeconds(start: compileStart, end: compileEnd)

        print("Library compilation (makeLibrary): \(String(format: "%.3f", compileTime * 1000)) ms")

        guard let function = compiledLibrary.makeFunction(name: "simple_add") else {
            print("Failed to make function")
            return
        }

        let pipelineStart = getTimeNanos()
        guard let pipelineState = try? device.makeComputePipelineState(function: function) else {
            print("Failed to create pipeline state")
            return
        }
        let pipelineEnd = getTimeNanos()
        let pipelineTime = getElapsedSeconds(start: pipelineStart, end: pipelineEnd)

        print("Pipeline state creation (makeComputePipelineState): \(String(format: "%.3f", pipelineTime * 1000)) ms")

        print("\n--- Kernel Launch Overhead ---")

        guard let inputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * workSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * workSize, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: workSize)
        for i in 0..<workSize {
            inputPtr[i] = Float(i)
        }

        // Warm-up run
        if let cmdBuffer = queue.makeCommandBuffer(),
           let encoder = cmdBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }

        var encodeTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cmdBuffer = queue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }

            let encodeStart = getTimeNanos()
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            let encodeEnd = getTimeNanos()
            encodeTimes.append(getElapsedSeconds(start: encodeStart, end: encodeEnd))
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }

        let avgEncodeTime = encodeTimes.reduce(0, +) / Double(iterations)
        print("Average encode time (dispatch): \(String(format: "%.3f", avgEncodeTime * 1e6)) μs")

        var execTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cmdBuffer = queue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)

            let execStart = getTimeNanos()
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            let execEnd = getTimeNanos()
            execTimes.append(getElapsedSeconds(start: execStart, end: execEnd))
        }

        let avgExecTime = execTimes.reduce(0, +) / Double(iterations)
        print("Average execution time (workSize=\(workSize)): \(String(format: "%.3f", avgExecTime * 1e6)) μs")

        let throughput = Double(workSize) / avgExecTime / 1e6
        print("Throughput: \(String(format: "%.2f", throughput)) M elements/s")

        print("\n--- Overhead Analysis ---")
        print("| Component | Time | Percentage of Execution |")
        print("|-----------|------|----------------------|")
        let overhead = avgEncodeTime / avgExecTime * 100
        print("| Encode overhead | \(String(format: "%.3f", avgEncodeTime * 1e6)) μs | \(String(format: "%.2f", overhead))% |")
        print("| Actual kernel | \(String(format: "%.3f", avgExecTime * 1e6)) μs | \(String(format: "%.2f", 100 - overhead))% |")

        print("\n--- Batch Dispatch Efficiency ---")
        let batchSizes = [1, 4, 16, 64]
        print("| Batch Size | Total Time | Per-Kernel | Speedup |")
        print("|------------|------------|------------|---------|")

        for batchSize in batchSizes {
            guard let cmdBuffer = queue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }

            let batchStart = getTimeNanos()
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            for _ in 0..<batchSize {
                encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            let batchEnd = getTimeNanos()
            let batchTime = getElapsedSeconds(start: batchStart, end: batchEnd)
            let perKernel = batchTime / Double(batchSize)
            let speedup = avgExecTime / perKernel
            print("| \(batchSize) | \(String(format: "%.3f", batchTime * 1e6)) μs | \(String(format: "%.3f", perKernel * 1e6)) μs | \(String(format: "%.2fx", speedup)) |")
        }

        print("\n--- Key Insights ---")
        print("1. Library compilation (makeLibrary) is expensive: \(String(format: "%.1f", compileTime * 1000)) ms")
        print("2. Pipeline state creation is moderate: \(String(format: "%.1f", pipelineTime * 1000)) ms")
        print("3. Encode/dispatch overhead is small but significant: \(String(format: "%.2f", overhead))%")
        print("4. Batching kernels reduces per-kernel overhead substantially")
        print("5. Precompiled shaders avoid runtime compilation cost")
        print("6. For short-running kernels, overhead can dominate execution time")
    }
}
