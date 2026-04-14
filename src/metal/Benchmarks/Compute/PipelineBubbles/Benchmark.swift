import Foundation
import Metal

// MARK: - Pipeline Bubbles and Instruction Latency Benchmark

let pipelineBubblesShaders = """
#include <metal_stdlib>
using namespace metal;

// Dependent chain: each op depends on previous result
kernel void dep_chain_add(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float v = in[id];
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    out[id] = v;
}

kernel void dep_chain_mul(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float v = in[id];
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    out[id] = v;
}

// Independent operations: can execute in parallel
kernel void indep_ops(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float v = in[id];
    float a = v + 1.0f;
    float b = v * 2.0f;
    float c = v + 2.0f;
    float d = v * 3.0f;
    out[id] = a + b + c + d;
}

// Mixed dependent and independent
kernel void mixed_pipe(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    float v = in[id];
    // Independent pairs
    float a = v + 1.0f;
    float b = v * 2.0f;
    float c = a + b;  // depends on a, b
    float d = a * b;  // depends on a, b
    out[id] = c + d;
}

// Deep dependency chain (causes pipeline bubbles)
kernel void deep_chain(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    float v = in[id];
    for (int i = 0; i < 10; i++) {
        v = v + 1.0f;
    }
    out[id] = v;
}

// Wide independent (no bubbles)
kernel void wide_indep(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    float v = in[id];
    float t0 = v + 1.0f;
    float t1 = v + 2.0f;
    float t2 = v + 3.0f;
    float t3 = v + 4.0f;
    float t4 = v + 5.0f;
    float t5 = v + 6.0f;
    float t6 = v + 7.0f;
    float t7 = v + 8.0f;
    out[id] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
}
"""

public struct PipelineBubblesBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Pipeline Bubbles and Instruction Latency")
        print("Measuring impact of dependent operations on GPU throughput")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: pipelineBubblesShaders, options: nil) else {
            print("Failed to create pipeline bubbles library")
            return
        }

        guard let depAddFunc = library.makeFunction(name: "dep_chain_add"),
              let depMulFunc = library.makeFunction(name: "dep_chain_mul"),
              let indepFunc = library.makeFunction(name: "indep_ops"),
              let mixedFunc = library.makeFunction(name: "mixed_pipe"),
              let deepFunc = library.makeFunction(name: "deep_chain"),
              let wideFunc = library.makeFunction(name: "wide_indep"),
              let depAddPipeline = try? device.makeComputePipelineState(function: depAddFunc),
              let depMulPipeline = try? device.makeComputePipelineState(function: depMulFunc),
              let indepPipeline = try? device.makeComputePipelineState(function: indepFunc),
              let mixedPipeline = try? device.makeComputePipelineState(function: mixedFunc),
              let deepPipeline = try? device.makeComputePipelineState(function: deepFunc),
              let widePipeline = try? device.makeComputePipelineState(function: wideFunc) else {
            print("Failed to create pipelines")
            return
        }

        let workSize = 65536
        let iterations = 100

        guard let inputBuffer = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: workSize)
        for i in 0..<workSize {
            inputPtr[i] = Float(i)
        }

        var sizeUInt = UInt32(workSize)
        let gridSize = MTLSize(width: workSize, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)

        print("\n--- Pipeline Efficiency Analysis ---")
        print("| Operation Type | Time (μs) | Throughput | Relative |")
        print("|----------------|-----------|------------|---------|")

        // Dependent Add Chain
        let depAddStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(depAddPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let depAddEnd = getTimeNanos()
        let depAddTime = getElapsedSeconds(start: depAddStart, end: depAddEnd) / Double(iterations)
        let depAddTP = Double(workSize) * Double(iterations) / depAddTime / 1e6

        // Dependent Mul Chain
        let depMulStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(depMulPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let depMulEnd = getTimeNanos()
        let depMulTime = getElapsedSeconds(start: depMulStart, end: depMulEnd) / Double(iterations)
        let depMulTP = Double(workSize) * Double(iterations) / depMulTime / 1e6

        // Independent Operations
        let indepStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(indepPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let indepEnd = getTimeNanos()
        let indepTime = getElapsedSeconds(start: indepStart, end: indepEnd) / Double(iterations)
        let indepTP = Double(workSize) * Double(iterations) / indepTime / 1e6

        // Mixed Pipeline
        let mixedStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(mixedPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let mixedEnd = getTimeNanos()
        let mixedTime = getElapsedSeconds(start: mixedStart, end: mixedEnd) / Double(iterations)
        let mixedTP = Double(workSize) * Double(iterations) / mixedTime / 1e6

        // Deep Chain
        let deepStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(deepPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let deepEnd = getTimeNanos()
        let deepTime = getElapsedSeconds(start: deepStart, end: deepEnd) / Double(iterations)
        let deepTP = Double(workSize) * Double(iterations) / deepTime / 1e6

        // Wide Independent
        let wideStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(widePipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let wideEnd = getTimeNanos()
        let wideTime = getElapsedSeconds(start: wideStart, end: wideEnd) / Double(iterations)
        let wideTP = Double(workSize) * Double(iterations) / wideTime / 1e6

        print("| Dep Add Chain | \(String(format: "%.2f", depAddTime * 1e6)) | \(String(format: "%.2f", depAddTP)) M/s | baseline |")
        print("| Dep Mul Chain | \(String(format: "%.2f", depMulTime * 1e6)) | \(String(format: "%.2f", depMulTP)) M/s | \(String(format: "%.2fx", depAddTP / depMulTP)) |")
        print("| Independent Ops | \(String(format: "%.2f", indepTime * 1e6)) | \(String(format: "%.2f", indepTP)) M/s | \(String(format: "%.2fx", depAddTP / indepTP)) |")
        print("| Mixed Pipeline | \(String(format: "%.2f", mixedTime * 1e6)) | \(String(format: "%.2f", mixedTP)) M/s | \(String(format: "%.2fx", depAddTP / mixedTP)) |")
        print("| Deep Chain (10x) | \(String(format: "%.2f", deepTime * 1e6)) | \(String(format: "%.2f", deepTP)) M/s | \(String(format: "%.2fx", depAddTP / deepTP)) |")
        print("| Wide Independent | \(String(format: "%.2f", wideTime * 1e6)) | \(String(format: "%.2f", wideTP)) M/s | \(String(format: "%.2fx", depAddTP / wideTP)) |")

        print("\n--- Key Insights ---")
        print("1. Dependent operations create pipeline bubbles, reducing throughput")
        print("2. Independent operations can execute in parallel, better utilize ALU")
        print("3. Deep dependency chains compound the bubble effect")
        print("4. Instruction-level parallelism (ILP) helps hide latency")
        print("5. Rearranging independent ops reduces pipeline stalls")
    }
}
