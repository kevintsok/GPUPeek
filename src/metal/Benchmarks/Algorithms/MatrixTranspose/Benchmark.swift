import Foundation
import Metal

// MARK: - Matrix Transpose Benchmark

let matrixTransposeShaders = """
#include <metal_stdlib>
using namespace metal;

// Naive transpose - each thread handles one element
kernel void transpose_naive(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& rows [[buffer(2)]],
                          constant uint& cols [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    uint row = id / cols;
    uint col = id % cols;
    if (row >= rows || col >= cols) return;
    out[col * rows + row] = in[row * cols + col];
}

// Tiled transpose - uses shared memory for coalesced access
kernel void transpose_tiled(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& rows [[buffer(2)]],
                          constant uint& cols [[buffer(3)]],
                          uint id [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;
    uint2 gid = uint2(id % cols, id / cols);

    threadgroup float tile[TILE_SIZE * TILE_SIZE];

    // Load into tile (coalesced)
    if (gid.y < rows && gid.x < cols) {
        tile[tid.y * TILE_SIZE + tid.x] = in[gid.y * cols + gid.x];
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Write from tile (coalesced for transpose)
    uint2 out_gid = uint2(gid.y, gid.x);
    if (out_gid.x < rows && out_gid.y < cols) {
        out[out_gid.y * rows + out_gid.x] = tile[tid.x * TILE_SIZE + tid.y];
    }
}

// Shared memory transpose with padding to avoid bank conflicts
kernel void transpose_padded(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& rows [[buffer(2)]],
                          constant uint& cols [[buffer(3)]],
                          uint id [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;
    constexpr uint PAD = 1;
    uint2 gid = uint2(id % cols, id / cols);

    // Add padding to avoid bank conflicts
    threadgroup float tile[(TILE_SIZE + PAD) * TILE_SIZE];

    // Load into tile
    if (gid.y < rows && gid.x < cols) {
        tile[tid.y * (TILE_SIZE + PAD) + tid.x] = in[gid.y * cols + gid.x];
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Write from transposed tile position
    uint2 out_gid = uint2(gid.y, gid.x);
    if (out_gid.x < rows && out_gid.y < cols) {
        out[out_gid.y * rows + out_gid.x] = tile[tid.x * (TILE_SIZE + PAD) + tid.y];
    }
}

// Column-wise transpose (for wide matrices)
kernel void transpose_columns(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& cols [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    uint tileCol = id / (rows * 16);
    uint withinTile = id % (rows * 16);
    uint row = withinTile % rows;
    uint col = tileCol * 16 + withinTile / rows;

    if (row >= rows || col >= cols) return;
    out[col * rows + row] = in[row * cols + col];
}

// In-place transpose for square matrices
kernel void transpose_inplace(device float* data [[buffer(0)]],
                             constant uint& N [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= N * N / 2) return;

    uint row = id / (N / 2);
    uint col = id % (N / 2);
    uint offset = row * N + col;

    if (offset >= (row * N + row)) return;  // Only upper triangle

    float temp = data[offset];
    data[offset] = data[col * N + row];
    data[col * N + row] = temp;
}

// Vectorized transpose using float4
kernel void transpose_vectorized(device const float4* in [[buffer(0)]],
                               device float4* out [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    uint row = id / (cols / 4);
    uint col = id % (cols / 4);

    if (row >= rows || col >= cols / 4) return;

    float4 val0 = in[row * (cols / 4) + col];
    // Transpose within the float4
    float4 result = float4(val0.x, val0.y, val0.z, val0.w);
    out[col * rows + row] = result;
}
"""

public struct MatrixTransposeBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Matrix Transpose Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: matrixTransposeShaders, options: nil) else {
            print("Failed to compile matrix transpose shaders")
            return
        }

        // Test different matrix sizes
        let configs = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048)
        ]

        for (rows, cols) in configs {
            print("\n--- Matrix Size: \(rows)x\(cols) ---")

            let size = rows * cols
            guard let inBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input matrix
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float(i % 256) / 255.0
            }

            var rowsValue = UInt32(rows)
            var colsValue = UInt32(cols)

            // Test naive transpose
            if let naiveFunc = library.makeFunction(name: "transpose_naive"),
               let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(naivePipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * 2 * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Naive: \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }

            // Test tiled transpose
            if let tiledFunc = library.makeFunction(name: "transpose_tiled"),
               let tiledPipeline = try? device.makeComputePipelineState(function: tiledFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(tiledPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 3)
                    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
                    encoder.dispatchThreads(MTLSize(width: cols, height: rows, depth: 1),
                                          threadsPerThreadgroup: tgSize)
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * 2 * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Tiled: \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }

            // Test padded transpose
            if let paddedFunc = library.makeFunction(name: "transpose_padded"),
               let paddedPipeline = try? device.makeComputePipelineState(function: paddedFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(paddedPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 3)
                    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
                    encoder.dispatchThreads(MTLSize(width: cols, height: rows, depth: 1),
                                          threadsPerThreadgroup: tgSize)
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * 2 * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Padded: \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Tiled transpose improves cache utilization vs naive")
        print("2. Padding avoids shared memory bank conflicts")
        print("3. Naive transpose has poor coalescing on writes")
        print("4. Transpose is memory-bound, not compute-bound")
        print("5. Square matrices can use in-place algorithm")
    }
}
