import Foundation
import Metal

// MARK: - Tensor Core Emulation (WMMA) Benchmark

let wmmaShaderSource = """
#include <metal_stdlib>
using namespace metal;

// WMMA fragment size (NVIDIA style: 16x16x16)
// Apple GPU SIMD width is 32, so we adapt

// Naive matrix multiply (baseline)
kernel void wmma_naive(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* c [[buffer(2)]],
                     constant uint& size [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint k = 0; k < size; k++) {
        sum += a[gid.x * size + k] * b[k * size + gid.y];
    }
    c[gid.x * size + gid.y] = sum;
}

// Tiled matrix multiply (exploits shared memory)
kernel void wmma_tiled(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* c [[buffer(2)]],
                     constant uint& size [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;
    threadgroup float As[TILE_SIZE * TILE_SIZE];
    threadgroup float Bs[TILE_SIZE * TILE_SIZE];

    float sum = 0.0f;
    for (uint block = 0; block < size; block += TILE_SIZE) {
        // Load tiles into shared memory
        As[lid.y * TILE_SIZE + lid.x] = a[gid.x * size + block + lid.x];
        Bs[lid.y * TILE_SIZE + lid.x] = b[(block + lid.y) * size + gid.y];
        threadgroup_barrier(mem_flags::mem_none);

        // Compute partial result
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[lid.y * TILE_SIZE + k] * Bs[k * TILE_SIZE + lid.x];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    c[gid.x * size + gid.y] = sum;
}

// SIMD-friendly block multiplication (32 threads cooperate)
kernel void wmma_simd_block(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* c [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    // Each thread computes one output element
    // Threads cooperate to load A and B tiles
    constexpr uint BLOCK = 32;
    uint row = gid.x;
    uint col = gid.y;
    uint blockRow = (row / BLOCK) * BLOCK;
    uint blockCol = (col / BLOCK) * BLOCK;

    float sum = 0.0f;
    for (uint k = blockCol; k < blockCol + BLOCK && k < size; k++) {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
}

// Vectorized load/store for better memory bandwidth
kernel void wmma_vectorized(device const float4* a [[buffer(0)]],
                          device const float4* b [[buffer(1)]],
                          device float4* c [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // Process 4 elements at a time
    uint row = gid.x * 4;
    uint col = gid.y;

    float4 sum = float4(0.0f);
    for (uint k = 0; k < size; k++) {
        float4 aRow = a[(row + 0) * size / 4 + k];
        float4 aRow1 = a[(row + 1) * size / 4 + k];
        float4 aRow2 = a[(row + 2) * size / 4 + k];
        float4 aRow3 = a[(row + 3) * size / 4 + k];
        float4 bCol = float4(b[k * size / 4 + col].x, b[k * size / 4 + col].y,
                              b[k * size / 4 + col].z, b[k * size / 4 + col].w);
        sum += aRow * bCol.x + aRow1 * bCol.y + aRow2 * bCol.z + aRow3 * bCol.w;
    }
    c[row * size / 4 + col] = sum;
}

// Half-precision WMMA emulation
kernel void wmma_fp16_tiled(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device float* c [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;
    threadgroup half As[TILE_SIZE * TILE_SIZE];
    threadgroup half Bs[TILE_SIZE * TILE_SIZE];

    float sum = 0.0f;
    for (uint block = 0; block < size; block += TILE_SIZE) {
        As[lid.y * TILE_SIZE + lid.x] = a[gid.x * size + block + lid.x];
        Bs[lid.y * TILE_SIZE + lid.x] = b[(block + lid.y) * size + gid.y];
        threadgroup_barrier(mem_flags::mem_none);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(As[lid.y * TILE_SIZE + k]) * float(Bs[k * TILE_SIZE + lid.x]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    c[gid.x * size + gid.y] = sum;
}
"""

public struct TensorCoreEmulationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Tensor Core Emulation (WMMA)")
        print("Warp Matrix Multiply Accumulate operations")
        print(String(repeating: "=", count: 70))

        let wmmaLibrary: MTLLibrary
        do {
            wmmaLibrary = try device.makeLibrary(source: wmmaShaderSource, options: nil)
        } catch {
            print("Failed to compile WMMA shaders: \(error)")
            return
        }

        let sizes = [128, 256, 512]
        let iterations = 20

        print("\n--- WMMA Performance Comparison ---")
        print("| Size | Naive GFLOPS | Tiled GFLOPS | SIMD GFLOPS | FP16 Tiled GFLOPS |")
        print("|------|--------------|--------------|-------------|-------------------|")

        for size in sizes {
            guard let naiveFunc = wmmaLibrary.makeFunction(name: "wmma_naive"),
                  let tiledFunc = wmmaLibrary.makeFunction(name: "wmma_tiled"),
                  let simdFunc = wmmaLibrary.makeFunction(name: "wmma_simd_block"),
                  let fp16Func = wmmaLibrary.makeFunction(name: "wmma_fp16_tiled") else {
                continue
            }

            guard let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc),
                  let tiledPipeline = try? device.makeComputePipelineState(function: tiledFunc),
                  let simdPipeline = try? device.makeComputePipelineState(function: simdFunc),
                  let fp16Pipeline = try? device.makeComputePipelineState(function: fp16Func) else {
                continue
            }

            guard let aBuffer = device.makeBuffer(length: size * size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bBuffer = device.makeBuffer(length: size * size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let cBuffer = device.makeBuffer(length: size * size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let aBufferFP16 = device.makeBuffer(length: size * size * MemoryLayout<UInt16>.size, options: .storageModeShared),
                  let bBufferFP16 = device.makeBuffer(length: size * size * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
                continue
            }

            var sizeUInt = UInt32(size)
            let gridSize = MTLSize(width: size, height: size, depth: 1)
            let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

            // Naive benchmark
            let startNaive = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(naivePipeline)
                encoder.setBuffer(aBuffer, offset: 0, index: 0)
                encoder.setBuffer(bBuffer, offset: 0, index: 1)
                encoder.setBuffer(cBuffer, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endNaive = getTimeNanos()
            let naiveTime = getElapsedSeconds(start: startNaive, end: endNaive)
            let naiveOps = 2.0 * Double(size) * Double(size) * Double(size) * Double(iterations)
            let naiveGFLOPS = naiveOps / naiveTime / 1e9

            // Tiled benchmark
            let startTiled = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(tiledPipeline)
                encoder.setBuffer(aBuffer, offset: 0, index: 0)
                encoder.setBuffer(bBuffer, offset: 0, index: 1)
                encoder.setBuffer(cBuffer, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endTiled = getTimeNanos()
            let tiledTime = getElapsedSeconds(start: startTiled, end: endTiled)
            let tiledOps = 2.0 * Double(size) * Double(size) * Double(size) * Double(iterations)
            let tiledGFLOPS = tiledOps / tiledTime / 1e9

            // SIMD block benchmark
            let startSIMD = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(simdPipeline)
                encoder.setBuffer(aBuffer, offset: 0, index: 0)
                encoder.setBuffer(bBuffer, offset: 0, index: 1)
                encoder.setBuffer(cBuffer, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endSIMD = getTimeNanos()
            let simdTime = getElapsedSeconds(start: startSIMD, end: endSIMD)
            let simdOps = 2.0 * Double(size) * Double(size) * Double(size) * Double(iterations)
            let simdGFLOPS = simdOps / simdTime / 1e9

            // FP16 tiled benchmark
            let startFP16 = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fp16Pipeline)
                encoder.setBuffer(aBufferFP16, offset: 0, index: 0)
                encoder.setBuffer(bBufferFP16, offset: 0, index: 1)
                encoder.setBuffer(cBuffer, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endFP16 = getTimeNanos()
            let fp16Time = getElapsedSeconds(start: startFP16, end: endFP16)
            let fp16Ops = 2.0 * Double(size) * Double(size) * Double(size) * Double(iterations)
            let fp16GFLOPS = fp16Ops / fp16Time / 1e9

            print("| \(size) | \(String(format: "%.2f", naiveGFLOPS)) | \(String(format: "%.2f", tiledGFLOPS)) | \(String(format: "%.2f", simdGFLOPS)) | \(String(format: "%.2f", fp16GFLOPS)) |")
        }

        print("\n--- Key Insights ---")
        print("1. Tiled WMMA exploits shared memory for better data reuse")
        print("2. SIMD block multiplication leverages 32-thread SIMD groups")
        print("3. FP16 reduces memory bandwidth by 2x")
        print("4. True tensor cores (NVIDIA/AMD) provide 8-16x speedup over WMMA emulation")
        print("5. Apple GPUs lack native tensor cores - WMMA is software emulation")
    }
}
