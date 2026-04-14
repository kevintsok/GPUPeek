import Foundation
import Metal

let scanShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void scan_kogge_stone(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    
    float val = in[id];
    
    float sum = val;
    sum += simd_shuffle_down(val, 1);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 16);
    
    out[id] = sum;
}

kernel void scan_hillis_steele(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    
    float val = in[id];
    
    for (uint offset = 1; offset < 32; offset *= 2) {
        float temp = simd_shuffle_up(val, offset);
        if (id >= offset) {
            val = val + temp;
        }
    }
    
    out[id] = val;
}
"""

public struct ScanBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Parallel Scan Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: scanShaders, options: nil) else {
            return
        }

        let size = 256 * 1024
        let iterations = 50

        guard let inBuf = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Scan Algorithms ---")

        if let koggeFunc = library.makeFunction(name: "scan_kogge_stone"),
           let koggePipeline = try? device.makeComputePipelineState(function: koggeFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(koggePipeline)
                encoder.setBuffer(inBuf, offset: 0, index: 0)
                encoder.setBuffer(outBuf, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Kogge-Stone: \(String(format: "%.3f", gops)) GOPS")
        }

        if let hillisFunc = library.makeFunction(name: "scan_hillis_steele"),
           let hillisPipeline = try? device.makeComputePipelineState(function: hillisFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(hillisPipeline)
                encoder.setBuffer(inBuf, offset: 0, index: 0)
                encoder.setBuffer(outBuf, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Hillis-Steele: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. Kogge-Stone is latency-optimal O(log n)")
        print("2. Hillis-Steele is work-efficient O(n log n)")
        print("3. SIMD shuffle enables efficient warp-level scan")
    }
}
EOF
echo "Created Scan/Benchmark.swift"