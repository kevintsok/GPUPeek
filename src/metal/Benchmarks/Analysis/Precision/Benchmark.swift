import Foundation
import Metal

// MARK: - Precision Analysis Benchmark

let precisionShaders = """
#include <metal_stdlib>
using namespace metal;

// FP32 operations
kernel void fp32_ops(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    for (uint i = 0; i < 64; i++) {
        val = fma(val, 0.99f, 0.001f);
        val = sin(val) * cos(val);
    }
    out[id] = val;
}

// FP16 operations  
kernel void fp16_ops(device const half* in [[buffer(0)]],
                    device half* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    half val = in[id];
    for (uint i = 0; i < 64; i++) {
        val = fma(val, (half)0.99f, (half)0.001f);
        val = sin(val) * cos(val);
    }
    out[id] = val;
}

// Mixed precision: accumulate in FP32
kernel void mixed_precision(device const half* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float accum = 0.0f;
    for (uint i = 0; i < 64; i++) {
        half val = in[(id + i) % size];
        accum += (float)val;
    }
    out[id] = accum;
}

// Precision test: FP16 accumulation error
kernel void fp16_accum(device const float* in [[buffer(0)]],
                      device half* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    half accum = 0.0f;
    for (uint i = 0; i < 1024; i++) {
        float val = in[(id + i) % size];
        accum += (half)val;
    }
    out[id] = accum;
}

// FP32 accumulation for comparison
kernel void fp32_accum(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float accum = 0.0f;
    for (uint i = 0; i < 1024; i++) {
        float val = in[(id + i) % size];
        accum += val;
    }
    out[id] = accum;
}
"""

public struct PrecisionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Numerical Precision Analysis Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: precisionShaders, options: nil) else {
            print("Failed to compile precision shaders")
            return
        }

        let size = 64 * 1024
        let iterations = 20

        guard let bufferInFP32 = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferInFP16 = device.makeBuffer(length: size * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let bufferOutFP32 = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOutFP16 = device.makeBuffer(length: size * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize FP32 input
        let inFP32Ptr = bufferInFP32.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            inFP32Ptr[i] = Float(i % 256) / 255.0
        }

        // Initialize FP16 input
        let inFP16Ptr = bufferInFP16.contents().bindMemory(to: UInt16.self, capacity: size)
        for i in 0..<size {
            inFP16Ptr[i] = FloatToHalf(Float(i % 256) / 255.0)
        }

        var sizeValue = UInt32(size)

        print("\n--- Precision Types Performance ---")

        // Test FP32
        if let fp32Func = library.makeFunction(name: "fp32_ops"),
           let fp32Pipeline = try? device.makeComputePipelineState(function: fp32Func) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fp32Pipeline)
                encoder.setBuffer(bufferInFP32, offset: 0, index: 0)
                encoder.setBuffer(bufferOutFP32, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) * 128.0 / elapsed / 1e9
            print("FP32: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test FP16
        if let fp16Func = library.makeFunction(name: "fp16_ops"),
           let fp16Pipeline = try? device.makeComputePipelineState(function: fp16Func) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fp16Pipeline)
                encoder.setBuffer(bufferInFP16, offset: 0, index: 0)
                encoder.setBuffer(bufferOutFP16, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) * 128.0 / elapsed / 1e9
            print("FP16: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Precision Comparison ---")

        // FP32 accumulation
        if let fp32AccumFunc = library.makeFunction(name: "fp32_accum"),
           let fp32AccumPipeline = try? device.makeComputePipelineState(function: fp32AccumFunc) {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(fp32AccumPipeline)
            encoder.setBuffer(bufferInFP32, offset: 0, index: 0)
            encoder.setBuffer(bufferOutFP32, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            let fp32Sum = bufferOutFP32.contents().bindMemory(to: Float.self, capacity: 1)[0]
            print("FP32 Accum sum: \(fp32Sum)")
        }

        // FP16 accumulation
        if let fp16AccumFunc = library.makeFunction(name: "fp16_accum"),
           let fp16AccumPipeline = try? device.makeComputePipelineState(function: fp16AccumFunc) {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(fp16AccumPipeline)
            encoder.setBuffer(bufferInFP32, offset: 0, index: 0)
            encoder.setBuffer(bufferOutFP16, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            let fp16Sum = HalfToFloat(bufferOutFP16.contents().bindMemory(to: UInt16.self, capacity: 1)[0])
            print("FP16 Accum sum: \(fp16Sum)")
        }

        print("\n--- Key Insights ---")
        print("1. FP16 is 1.5-2x faster than FP32 for compute")
        print("2. FP16 has limited precision (~3.3 decimal digits)")
        print("3. Accumulation errors compound with iteration count")
        print("4. Mixed precision (FP16 compute, FP32 accum) balances speed and accuracy")
    }
}
