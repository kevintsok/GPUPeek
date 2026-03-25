import Foundation
import Metal

let convolutionShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void conv3x3_naive(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint2& size [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = int(gid.x) + dx;
            int sy = int(gid.y) + dy;
            sx = max(0, min(int(size.x) - 1, sx));
            sy = max(0, min(int(size.y) - 1, sy));
            sum += in[sy * int(size.x) + sx];
        }
    }
    out[gid.y * size.x + gid.x] = sum * 0.111f;
}

kernel void conv3x3_shared(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          threadgroup float* shared [[threadgroup(0)]],
                          constant uint2& size [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    
    shared[lid.y * 18 + lid.x] = in[gid.y * size.x + gid.x];
    threadgroup_barrier(flags::mem_threadgroup);
    
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int lx = int(lid.x) + dx + 1;
            int ly = int(lid.y) + dy + 1;
            sum += shared[ly * 18 + lx];
        }
    }
    out[gid.y * size.x + gid.x] = sum * 0.111f;
}
"""

public struct ConvolutionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Convolution Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: convolutionShaders, options: nil) else {
            return
        }

        let width = 1024
        let height = 1024
        let iterations = 20

        guard let inBuf = device.makeBuffer(length: width * height * MemoryLayout<Float>.size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: width * height * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeValue = SIMD2<UInt32>(UInt32(width), UInt32(height))

        print("\n--- 3x3 Convolution ---")

        if let naiveFunc = library.makeFunction(name: "conv3x3_naive"),
           let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(naivePipeline)
                encoder.setBuffer(inBuf, offset: 0, index: 0)
                encoder.setBuffer(outBuf, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(width * height) / elapsed / 1e9
            print("Naive: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. Convolution is memory-intensive (9 reads per pixel)")
        print("2. Shared memory optimization helps reduce global memory access")
        print("3. 3x3 convolution is common in CNNs")
    }
}
EOF
echo "Created Convolution/Benchmark.swift"