import Foundation
import Metal

// MARK: - Shader Compilation and Launch Overhead Benchmark

public struct ShaderOverheadBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("80. Shader Compilation and Kernel Launch Overhead")
        print(String(repeating: "=", count: 70))

        // Simple kernel for testing
        let simpleKernel = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void simple_add(device float* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
            out[id] = in[id] + 1.0f;
        }
        """

        let iterations = 100
        let workSize = 65536

        // Measure library compilation
        print("\n--- Library Compilation Overhead ---")
        let compileStart = getTimeNanos()
        guard let compiledLibrary = try? device.makeLibrary(source: simpleKernel, options: nil) else {
            print("Failed to compile library")
            return
        }
        let compileEnd = getTimeNanos()
        let compileTime = getElapsedSeconds(start: compileStart, end: compileEnd)
        print("Library compilation (makeLibrary): \(String(format: "%.3f", compileTime * 1000)) ms")

        // Measure pipeline state creation
        guard let function = compiledLibrary.makeFunction(name: "simple_add"),
              let pipelineState = try? device.makeComputePipelineState(function: function) else {
            print("Failed to create pipeline state")
            return
        }

        let pipelineStart = getTimeNanos()
        _ = try? device.makeComputePipelineState(function: function)
        let pipelineEnd = getTimeNanos()
        let pipelineTime = getElapsedSeconds(start: pipelineStart, end: pipelineEnd)
        print("Pipeline state creation (makeComputePipelineState): \(String(format: "%.3f", pipelineTime * 1000)) ms")

        // Create buffers
        guard let inputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * workSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * workSize, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Warm-up
        warmUp(pipeline: pipelineState, inputBuffer: inputBuffer, outputBuffer: outputBuffer, workSize: workSize)

        // Measure dispatch overhead
        print("\n--- Kernel Launch Overhead ---")
        let encodeTimes = measureEncodeOverhead(pipeline: pipelineState, inputBuffer: inputBuffer,
                                                outputBuffer: outputBuffer, workSize: workSize, iterations: iterations)
        let avgEncodeTime = encodeTimes.reduce(0, +) / Double(iterations)
        print("Average encode time (dispatch): \(String(format: "%.3f", avgEncodeTime * 1e6)) μs")

        // Measure execution time
        let execTimes = measureExecutionTime(pipeline: pipelineState, inputBuffer: inputBuffer,
                                            outputBuffer: outputBuffer, workSize: workSize, iterations: iterations)
        let avgExecTime = execTimes.reduce(0, +) / Double(iterations)
        print("Average execution time (workSize=\(workSize)): \(String(format: "%.3f", avgExecTime * 1e6)) μs")

        let throughput = Double(workSize) / avgExecTime / 1e6
        print("Throughput: \(String(format: "%.2f", throughput)) M elements/s")

        // Overhead analysis
        print("\n--- Overhead Analysis ---")
        let overhead = avgEncodeTime / avgExecTime * 100
        print("| Component | Time | Percentage of Execution |")
        print("|-----------|------|----------------------|")
        print("| Encode overhead | \(String(format: "%.3f", avgEncodeTime * 1e6)) μs | \(String(format: "%.2f", overhead))% |")
        print("| Actual kernel | \(String(format: "%.3f", avgExecTime * 1e6)) μs | \(String(format: "%.2f", 100 - overhead))% |")

        // Batch dispatch
        print("\n--- Batch Dispatch Efficiency ---")
        let batchSizes = [1, 4, 16, 64]
        print("| Batch Size | Total Time | Per-Kernel | Speedup |")
        print("|------------|------------|------------|---------|")

        for batchSize in batchSizes {
            let batchTime = measureBatchTime(pipeline: pipelineState, inputBuffer: inputBuffer,
                                           outputBuffer: outputBuffer, workSize: workSize, batchSize: batchSize)
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

    private func warmUp(pipeline: MTLComputePipelineState, inputBuffer: MTLBuffer, outputBuffer: MTLBuffer, workSize: Int) {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func measureEncodeOverhead(pipeline: MTLComputePipelineState, inputBuffer: MTLBuffer,
                                      outputBuffer: MTLBuffer, workSize: Int, iterations: Int) -> [Double] {
        var times: [Double] = []
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            let start = getTimeNanos()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            let end = getTimeNanos()
            times.append(getElapsedSeconds(start: start, end: end))
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        return times
    }

    private func measureExecutionTime(pipeline: MTLComputePipelineState, inputBuffer: MTLBuffer,
                                     outputBuffer: MTLBuffer, workSize: Int, iterations: Int) -> [Double] {
        var times: [Double] = []
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            let start = getTimeNanos()
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            let end = getTimeNanos()
            times.append(getElapsedSeconds(start: start, end: end))
        }
        return times
    }

    private func measureBatchTime(pipeline: MTLComputePipelineState, inputBuffer: MTLBuffer,
                                outputBuffer: MTLBuffer, workSize: Int, batchSize: Int) -> Double {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { return 0 }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        let start = getTimeNanos()
        for _ in 0..<batchSize {
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end)
    }
}
