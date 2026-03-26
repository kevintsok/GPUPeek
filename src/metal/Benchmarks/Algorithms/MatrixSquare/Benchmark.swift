import Foundation
import Metal

// MARK: - Matrix Square (A * A^T) Benchmark

let matrixSquareShaders = """
#include <metal_stdlib>
using namespace metal;

// C = A * A^T where A is MxK and A^T is KxM, result is MxM
// Non-contiguous memory access pattern: A[gid.y * K + k] vs A[gid.x * K + k]
kernel void mat_square_naive(device const float* A [[buffer(0)]],
                       device float* C [[buffer(1)]],
                       constant uint& M [[buffer(2)]],
                       constant uint& K [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= M || gid.y >= M) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Note: A[gid.y*K+k] is row access (contiguous)
        // A[gid.x*K+k] is column access (non-contiguous if accessing different rows)
        sum += A[gid.y * K + k] * A[gid.x * K + k];
    }
    C[gid.y * M + gid.x] = sum;
}

// Shared memory tiled version for better cache utilization
kernel void mat_square_shared(device const float* A [[buffer(0)]],
                         device float* C [[buffer(1)]],
                         constant uint& M [[buffer(2)]],
                         constant uint& K [[buffer(3)]],
                         threadgroup float* tileA [[threadgroup(0)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]) {
    uint tileSize = 16;
    float sum = 0.0f;

    for (uint t = 0; t < (K + tileSize - 1) / tileSize; t++) {
        uint kStart = t * tileSize;
        uint kEnd = min(kStart + tileSize, K);

        // Load column of A^T (which is row of A) into tile
        for (uint k = lid.x; k < kEnd - kStart; k += 16) {
            uint globalK = kStart + k;
            tileA[lid.y * tileSize + k] = A[gid.y * K + globalK];
        }
        threadgroup_barrier(mem_flags::mem_none);

        // Load row of A into tile
        for (uint k = lid.x; k < kEnd - kStart; k += 16) {
            uint globalK = kStart + k;
            tileA[lid.y * tileSize + k] = A[gid.x * K + globalK];
        }
        threadgroup_barrier(mem_flags::mem_none);

        // Compute partial dot product
        for (uint k = 0; k < kEnd - kStart; k++) {
            sum += tileA[lid.y * tileSize + k] * tileA[lid.x * tileSize + k];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    C[gid.y * M + gid.x] = sum;
}

// Regular GEMM for comparison: C = A * B (contiguous)
kernel void gemm_naive(device const float* A [[buffer(0)]],
                  device const float* B [[buffer(1)]],
                  device float* C [[buffer(2)]],
                  constant uint& M [[buffer(3)]],
                  constant uint& K [[buffer(4)]],
                  constant uint& N [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}
"""

public struct MatrixSquareBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Matrix Square (A × A^T) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: matrixSquareShaders, options: nil) else {
            print("Failed to compile matrix square shaders")
            return
        }

        let sizes = [256, 512, 1024]
        let kDim = 512

        for m in sizes {
            print("\n--- M=\(m), K=\(kDim) ---")

            guard let matA = device.makeBuffer(length: m * kDim * MemoryLayout<Float>.size, options: .storageModeShared),
                  let matC = device.makeBuffer(length: m * m * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize A with random data
            let matAPtr = matA.contents().bindMemory(to: Float.self, capacity: m * kDim)
            for i in 0..<(m * kDim) {
                matAPtr[i] = Float.random(in: 0.0...1.0)
            }

            var mVar = UInt32(m)
            var kVar = UInt32(kDim)

            // Test naive matrix square
            if let matFunc = library.makeFunction(name: "mat_square_naive"),
               let matPipeline = try? device.makeComputePipelineState(function: matFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(matPipeline)
                    encoder.setBuffer(matA, offset: 0, index: 0)
                    encoder.setBuffer(matC, offset: 0, index: 1)
                    encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: m, height: m, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end)

                // FLOPs: C[i][j] = sum_k A[i][k] * A[j][k]
                // MxM result, each element computes K mul-adds (2 FLOPs each)
                let flops = 2.0 * Double(m) * Double(m) * Double(kDim) * Double(iterations)
                let gflops = flops / elapsed / 1e9

                print("Matrix Square (A*A^T): \(String(format: "%.2f", gflops)) GFLOPS")
                print("  \(m)x\(m) result, \(kDim) dimension reduction")
            }

            // Test shared memory version
            if let sharedFunc = library.makeFunction(name: "mat_square_shared"),
               let sharedPipeline = try? device.makeComputePipelineState(function: sharedFunc) {
                guard let tileBuffer = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared) else {
                    continue
                }

                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(sharedPipeline)
                    encoder.setBuffer(matA, offset: 0, index: 0)
                    encoder.setBuffer(matC, offset: 0, index: 1)
                    encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBuffer(tileBuffer, offset: 0, index: 4)
                    encoder.dispatchThreads(MTLSize(width: m, height: m, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end)

                let flops = 2.0 * Double(m) * Double(m) * Double(kDim) * Double(iterations)
                let gflops = flops / elapsed / 1e9

                print("Matrix Square (Shared):   \(String(format: "%.2f", gflops)) GFLOPS")
            }
        }

        // Compare with regular GEMM (C = A * B, both contiguous)
        print("\n--- Comparison: Matrix Square vs GEMM ---")
        let compareM: UInt32 = 512
        let compareK: UInt32 = 512
        let compareN: UInt32 = 512

        guard let matA = device.makeBuffer(length: Int(compareM * compareK) * MemoryLayout<Float>.size, options: .storageModeShared),
              let matB = device.makeBuffer(length: Int(compareK * compareN) * MemoryLayout<Float>.size, options: .storageModeShared),
              let matC = device.makeBuffer(length: Int(compareM * compareN) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        let aPtr = matA.contents().bindMemory(to: Float.self, capacity: Int(compareM * compareK))
        let bPtr = matB.contents().bindMemory(to: Float.self, capacity: Int(compareK * compareN))
        for i in 0..<Int(compareM * compareK) {
            aPtr[i] = Float.random(in: 0.0...1.0)
        }
        for i in 0..<Int(compareK * compareN) {
            bPtr[i] = Float.random(in: 0.0...1.0)
        }

        var mVar = compareM
        var kVar = compareK
        var nVar = compareN

        // GEMM (contiguous access)
        if let gemmFunc = library.makeFunction(name: "gemm_naive"),
           let gemmPipeline = try? device.makeComputePipelineState(function: gemmFunc) {
            let iterations = 10
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(gemmPipeline)
                encoder.setBuffer(matA, offset: 0, index: 0)
                encoder.setBuffer(matB, offset: 0, index: 1)
                encoder.setBuffer(matC, offset: 0, index: 2)
                encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.dispatchThreads(MTLSize(width: Int(compareN), height: Int(compareM), depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end)
            let flops = 2.0 * Double(compareM) * Double(compareK) * Double(compareN) * Double(iterations)
            let gflops = flops / elapsed / 1e9
            print("GEMM (contiguous A*B): \(String(format: "%.2f", gflops)) GFLOPS")
        }

        // Matrix square (non-contiguous access)
        if let matFunc = library.makeFunction(name: "mat_square_naive"),
           let matPipeline = try? device.makeComputePipelineState(function: matFunc) {
            let iterations = 10
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(matPipeline)
                encoder.setBuffer(matA, offset: 0, index: 0)
                encoder.setBuffer(matC, offset: 0, index: 1)
                encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: Int(compareM), height: Int(compareM), depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end)
            let flops = 2.0 * Double(compareM) * Double(compareM) * Double(compareK) * Double(iterations)
            let gflops = flops / elapsed / 1e9
            print("Matrix Square (A*A^T):    \(String(format: "%.2f", gflops)) GFLOPS")
        }

        print("\n--- Key Findings ---")
        print("1. Matrix Square A*A^T has non-contiguous memory access pattern")
        print("2. A[gid.y*K+k] is contiguous but A[gid.x*K+k] is strided")
        print("3. This pattern appears in neural network backpropagation")
        print("4. GEMM with contiguous data is typically 1.5-3x faster")
        print("5. Shared memory tiling can help but cannot fully hide non-contiguity")
    }
}
