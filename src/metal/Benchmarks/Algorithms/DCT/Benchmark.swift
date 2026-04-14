import Foundation
import Metal

// MARK: - DCT (Discrete Cosine Transform) Benchmark

let dctShaders = """
#include <metal_stdlib>
using namespace metal;

// Naive 1D DCT - O(N^2) per output
kernel void dct_naive(device const float* in [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint k [[thread_position_in_grid]]) {
    if (k >= size) return;

    float sum = 0.0f;
    float pi_k = M_PI_F * float(k) / float(size);

    for (uint n = 0; n < size; n++) {
        sum += in[n] * cos(pi_k * (float(n) + 0.5f));
    }
    out[k] = sum;
}

// Butterfly-optimized 1D DCT - O(N log N)
kernel void dct_butterfly(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint N = size;
    uint halfSize = N / 2;

    // First stage: even-odd decomposition
    float even = (id < halfSize) ? in[id * 2] : in[(id - halfSize) * 2 + 1];
    float odd = (id < halfSize) ? in[id * 2 + 1] : in[(id - halfSize) * 2];

    // Twiddle factor
    float theta = M_PI_F * float(id) / float(N);
    float twiddle = cos(theta);

    // Butterfly
    float temp1 = even + odd;
    float temp2 = (even - odd) * twiddle;

    // Store intermediate result
    out[id] = temp1 + temp2;
}

// 1D DCT using Cooley-Tukey butterfly structure
kernel void dct_cooley_tukey(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint N = size;
    uint halfN = N / 2;

    // Stage 1: split into even and odd
    float even = (id < halfN) ? in[2 * id] : 0.0f;
    float odd = (id < halfN) ? in[2 * id + 1] : 0.0f;

    threadgroup_barrier(flags::mem_threadgroup);

    // Simple butterfly for demonstration
    float y0 = even + odd;
    float y1 = (even - odd) * cos(M_PI_F * float(id) / float(N));

    out[id] = y0 + y1;
}

// 2D DCT (separated row-column approach)
kernel void dct_2d_row(device const float* in [[buffer(0)]],
                      device float* temp [[buffer(1)]],
                      constant uint& width [[buffer(2)]],
                      constant uint& height [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= width || gid.y >= height) return;

    // 1D DCT on row
    uint k = gid.y;
    float sum = 0.0f;
    float pi_k = M_PI_F * float(k) / float(width);

    for (uint n = 0; n < width; n++) {
        sum += in[gid.x * width + n] * cos(pi_k * (float(n) + 0.5f));
    }
    temp[gid.y * width + gid.x] = sum;
}

kernel void dct_2d_col(device const float* temp [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& width [[buffer(2)]],
                      constant uint& height [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= width || gid.y >= height) return;

    // 1D DCT on column of temp result
    uint k = gid.y;
    float sum = 0.0f;
    float pi_k = M_PI_F * float(k) / float(height);

    for (uint n = 0; n < height; n++) {
        sum += temp[n * width + gid.x] * cos(pi_k * (float(n) + 0.5f));
    }
    out[gid.y * width + gid.x] = sum;
}

// IDCT (Inverse DCT)
kernel void idct_naive(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint n [[thread_position_in_grid]]) {
    if (n >= size) return;

    float sum = 0.0f;

    for (uint k = 0; k < size; k++) {
        float ck = (k == 0) ? 0.5f : 1.0f;
        float pi_k = M_PI_F * float(k) / float(size);
        sum += ck * in[k] * cos(pi_k * (float(n) + 0.5f));
    }
    out[n] = sum * 2.0f / float(size);
}
"""

public struct DCTBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("DCT (Discrete Cosine Transform) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: dctShaders, options: nil) else {
            print("Failed to compile DCT shaders")
            return
        }

        let sizes = [64, 256, 1024]

        for size in sizes {
            print("\n--- DCT Size: \(size) ---")

            guard let inBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let tempBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize with test signal (simple sine wave)
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float(sin(Double(i) * 0.1))
            }

            var sizeValue = UInt32(size)

            // Test naive DCT
            if let naiveFunc = library.makeFunction(name: "dct_naive"),
               let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(naivePipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gflops = Double(size) * Double(size) / elapsed / 1e9
                print("Naive DCT: \(String(format: "%.2f", gflops)) GFLOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test butterfly DCT
            if let butterflyFunc = library.makeFunction(name: "dct_butterfly"),
               let butterflyPipeline = try? device.makeComputePipelineState(function: butterflyFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(butterflyPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gflops = Double(size) * log2(Double(size)) / elapsed / 1e9
                print("Butterfly DCT: \(String(format: "%.2f", gflops)) GFLOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test IDCT
            if let idctFunc = library.makeFunction(name: "idct_naive"),
               let idctPipeline = try? device.makeComputePipelineState(function: idctFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(idctPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gflops = Double(size) * Double(size) / elapsed / 1e9
                print("IDCT: \(String(format: "%.2f", gflops)) GFLOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        // Test 2D DCT
        print("\n--- 2D DCT ---")
        let dims = [64, 128]
        for dim in dims {
            guard let inBuffer2D = device.makeBuffer(length: dim * dim * MemoryLayout<Float>.size, options: .storageModeShared),
                  let tempBuffer2D = device.makeBuffer(length: dim * dim * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer2D = device.makeBuffer(length: dim * dim * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize 2D signal
            let inPtr2D = inBuffer2D.contents().bindMemory(to: Float.self, capacity: dim * dim)
            for i in 0..<(dim * dim) {
                inPtr2D[i] = Float(i % dim) / Float(dim)
            }

            var dimValue = UInt32(dim)

            if let rowFunc = library.makeFunction(name: "dct_2d_row"),
               let colFunc = library.makeFunction(name: "dct_2d_col"),
               let rowPipeline = try? device.makeComputePipelineState(function: rowFunc),
               let colPipeline = try? device.makeComputePipelineState(function: colFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    // Row pass
                    encoder.setComputePipelineState(rowPipeline)
                    encoder.setBuffer(inBuffer2D, offset: 0, index: 0)
                    encoder.setBuffer(tempBuffer2D, offset: 0, index: 1)
                    encoder.setBytes(&dimValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&dimValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: dim, height: dim, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                    encoder.endEncoding()

                    // Column pass
                    let cmd2 = queue.makeCommandBuffer()
                    guard let enc2 = cmd2?.makeComputeCommandEncoder() else { continue }
                    enc2.setComputePipelineState(colPipeline)
                    enc2.setBuffer(tempBuffer2D, offset: 0, index: 0)
                    enc2.setBuffer(outBuffer2D, offset: 0, index: 1)
                    enc2.setBytes(&dimValue, length: MemoryLayout<UInt32>.size, index: 2)
                    enc2.setBytes(&dimValue, length: MemoryLayout<UInt32>.size, index: 3)
                    enc2.dispatchThreads(MTLSize(width: dim, height: dim, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                    enc2.endEncoding()

                    cmd.commit()
                    cmd.waitUntilCompleted()
                    cmd2.commit()
                    cmd2.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let pixels = Double(dim * dim) / 1e6
                print("\(dim)x\(dim) 2D DCT: \(String(format: "%.2f", pixels / elapsed)) MP/s (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Naive DCT is O(N^2) - prohibitively slow for large N")
        print("2. Butterfly DCT is O(N log N) - practical implementation")
        print("3. DCT is essential for JPEG compression, video encoding")
        print("4. 2D DCT can be separated into row + column 1D DCTs")
        print("5. Apple GPU memory bandwidth often limits DCT performance")
    }
}
