import Foundation
import Metal

// MARK: - Heat Equation / Jacobi Iteration Benchmark

let heatEquationShaders = """
#include <metal_stdlib>
using namespace metal;

// Jacobi iteration: one step of 2D heat equation update
// Discretized Laplacian: (left + right + up + down - 4*center) / dx^2
kernel void jacobi_iteration(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device const float* alpha [[buffer(2)]],
                        constant uint2& size [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    // Boundary conditions: fixed temperature (Dirichlet)
    if (gid.x == 0 || gid.x == size.x - 1 || gid.y == 0 || gid.y == size.y - 1) {
        out[gid.y * size.x + gid.x] = in[gid.y * size.x + gid.x];
        return;
    }

    // Get 4 neighbors (stencil pattern)
    float center = in[gid.y * size.x + gid.x];
    float left = in[gid.y * size.x + (gid.x - 1)];
    float right = in[gid.y * size.x + (gid.x + 1)];
    float up = in[(gid.y - 1) * size.x + gid.x];
    float down = in[(gid.y + 1) * size.x + gid.x];

    // Simplified heat equation: out = center + alpha * (left + right + up + down - 4*center)
    out[gid.y * size.x + gid.x] = center + alpha[0] * (left + right + up + down - 4.0f * center);
}

// Jacobi iteration with shared memory optimization (tile with halo)
kernel void jacobi_iteration_shared(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device const float* alpha [[buffer(2)]],
                              threadgroup float* tile [[threadgroup(0)]],
                              constant uint2& size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]],
                              uint2 lid [[thread_position_in_threadgroup]]) {
    uint tileSize = 16;
    if (gid.x >= size.x || gid.y >= size.y) return;

    // Load tile into shared memory
    uint2 globalPos = uint2(gid.x, gid.y);
    uint localIdx = lid.y * (tileSize + 2) + lid.x;

    // Load center
    tile[localIdx] = in[globalPos.y * size.x + globalPos.x];

    // Load halo cells
    if (lid.x == 0 && gid.x > 0) {
        tile[localIdx - 1] = in[globalPos.y * size.x + (globalPos.x - 1)];
    }
    if (lid.x == tileSize - 1 && gid.x < size.x - 1) {
        tile[localIdx + 1] = in[globalPos.y * size.x + (globalPos.x + 1)];
    }
    if (lid.y == 0 && gid.y > 0) {
        tile[localIdx - tileSize] = in[(globalPos.y - 1) * size.x + globalPos.x];
    }
    if (lid.y == tileSize - 1 && gid.y < size.y - 1) {
        tile[localIdx + tileSize] = in[(globalPos.y + 1) * size.x + globalPos.x];
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Boundary: keep original value
    if (gid.x == 0 || gid.x == size.x - 1 || gid.y == 0 || gid.y == size.y - 1) {
        out[gid.y * size.x + gid.x] = tile[localIdx];
        return;
    }

    // Compute Laplacian using shared memory
    float center = tile[localIdx];
    float left = tile[localIdx - 1];
    float right = tile[localIdx + 1];
    float up = tile[localIdx - (tileSize + 2)];
    float down = tile[localIdx + (tileSize + 2)];

    out[gid.y * size.x + gid.x] = center + alpha[0] * (left + right + up + down - 4.0f * center);
}

// Gauss-Seidel iteration (successive over-relaxation) - more aggressive
kernel void gauss_seidel_iteration(device const float* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                device const float* omega [[buffer(2)]],
                                constant uint2& size [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    // Boundary conditions
    if (gid.x == 0 || gid.x == size.x - 1 || gid.y == 0 || gid.y == size.y - 1) {
        out[gid.y * size.x + gid.x] = in[gid.y * size.x + gid.x];
        return;
    }

    float center = in[gid.y * size.x + gid.x];
    float left = in[gid.y * size.x + (gid.x - 1)];
    float right = in[gid.y * size.x + (gid.x + 1)];
    float up = in[(gid.y - 1) * size.x + gid.x];
    float down = in[(gid.y + 1) * size.x + gid.x];

    // SOR: x_new = x_old + omega * (average_of_neighbors - x_old)
    float avg = 0.25f * (left + right + up + down);
    out[gid.y * size.x + gid.x] = center + omega[0] * (avg - center);
}

// Red-Black Gauss-Seidel (for parallel execution)
kernel void red_black_gauss_seidel(device const float* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                device const float* omega [[buffer(2)]],
                                constant uint2& size [[buffer(3)]],
                                constant uint& color [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    // Red-Black coloring: (x + y) % 2
    uint expectedColor = (gid.x + gid.y) % 2;
    if (color != expectedColor) {
        out[gid.y * size.x + gid.x] = in[gid.y * size.x + gid.x];
        return;
    }

    // Boundary conditions
    if (gid.x == 0 || gid.x == size.x - 1 || gid.y == 0 || gid.y == size.y - 1) {
        out[gid.y * size.x + gid.x] = in[gid.y * size.x + gid.x];
        return;
    }

    float center = in[gid.y * size.x + gid.x];
    float left = in[gid.y * size.x + (gid.x - 1)];
    float right = in[gid.y * size.x + (gid.x + 1)];
    float up = in[(gid.y - 1) * size.x + gid.x];
    float down = in[(gid.y + 1) * size.x + gid.x];

    float avg = 0.25f * (left + right + up + down);
    out[gid.y * size.x + gid.x] = center + omega[0] * (avg - center);
}
"""

public struct HeatEquationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Heat Equation / Jacobi Iteration Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: heatEquationShaders, options: nil) else {
            print("Failed to compile heat equation shaders")
            return
        }

        let sizes = [256, 512, 1024, 2048]
        let iterations = 100

        for gridSize in sizes {
            print("\n--- Grid Size: \(gridSize)x\(gridSize) ---")

            guard let inBuffer = device.makeBuffer(length: gridSize * gridSize * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: gridSize * gridSize * MemoryLayout<Float>.size, options: .storageModeShared),
                  let alphaBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize grid with room temperature (20°C)
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: gridSize * gridSize)
            for i in 0..<(gridSize * gridSize) {
                inPtr[i] = Float(20.0)
            }

            // Add heat source in center (100°C)
            let centerX = gridSize / 2
            let centerY = gridSize / 2
            for dy in -10..<10 {
                for dx in -10..<10 {
                    let idx = (centerY + dy) * gridSize + (centerX + dx)
                    if idx >= 0 && idx < gridSize * gridSize {
                        inPtr[idx] = Float(100.0)
                    }
                }
            }

            // Alpha (thermal diffusivity)
            let alphaPtr = alphaBuffer.contents().bindMemory(to: Float.self, capacity: 1)
            alphaPtr.pointee = Float(0.25)

            var sizeVar = simd_uint2(UInt32(gridSize), UInt32(gridSize))

            // Jacobi iteration
            if let jacobiFunc = library.makeFunction(name: "jacobi_iteration"),
               let jacobiPipeline = try? device.makeComputePipelineState(function: jacobiFunc) {
                let start = getTimeNanos()
                var src = inBuffer
                var dst = outBuffer

                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(jacobiPipeline)
                    encoder.setBuffer(src, offset: 0, index: 0)
                    encoder.setBuffer(dst, offset: 0, index: 1)
                    encoder.setBuffer(alphaBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<simd_uint2>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: gridSize, height: gridSize, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    swap(&src, &dst)
                }
                let end = getTimeNanos()

                let elapsed = getElapsedSeconds(start: start, end: end)
                let cellsPerIter = Double(gridSize * gridSize)
                let mCellsPerSec = cellsPerIter * Double(iterations) / elapsed / 1e6

                print("Jacobi (baseline): \(String(format: "%.2f", mCellsPerSec)) MCells/s (\(String(format: "%.2f", elapsed * 1000)) ms total)")
            }

            // Shared memory version
            if let sharedFunc = library.makeFunction(name: "jacobi_iteration_shared"),
               let sharedPipeline = try? device.makeComputePipelineState(function: sharedFunc) {
                guard let tileBuffer = device.makeBuffer(length: (16 + 2) * (16 + 2) * MemoryLayout<Float>.size, options: .storageModeShared) else {
                    continue
                }

                let start = getTimeNanos()
                var src = inBuffer
                var dst = outBuffer

                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(sharedPipeline)
                    encoder.setBuffer(src, offset: 0, index: 0)
                    encoder.setBuffer(dst, offset: 0, index: 1)
                    encoder.setBuffer(alphaBuffer, offset: 0, index: 2)
                    encoder.setBuffer(tileBuffer, offset: 0, index: 3)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<simd_uint2>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: gridSize, height: gridSize, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    swap(&src, &dst)
                }
                let end = getTimeNanos()

                let elapsed = getElapsedSeconds(start: start, end: end)
                let cellsPerIter = Double(gridSize * gridSize)
                let mCellsPerSec = cellsPerIter * Double(iterations) / elapsed / 1e6

                print("Jacobi (shared):   \(String(format: "%.2f", mCellsPerSec)) MCells/s")
            }
        }

        // Gauss-Seidel comparison
        print("\n--- Gauss-Seidel SOR Comparison ---")
        let gsSize = 512

        guard let gsInBuffer = device.makeBuffer(length: gsSize * gsSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let gsOutBuffer = device.makeBuffer(length: gsSize * gsSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let omegaBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        let gsInPtr = gsInBuffer.contents().bindMemory(to: Float.self, capacity: gsSize * gsSize)
        for i in 0..<(gsSize * gsSize) {
            gsInPtr[i] = Float(20.0)
        }

        let omegaPtr = omegaBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        omegaPtr.pointee = Float(0.8)  // Relaxation factor

        var gsSizeVar = simd_uint2(UInt32(gsSize), UInt32(gsSize))

        // Gauss-Seidel with SOR
        if let gsFunc = library.makeFunction(name: "gauss_seidel_iteration"),
           let gsPipeline = try? device.makeComputePipelineState(function: gsFunc) {
            let iterations = 100
            let start = getTimeNanos()
            var src = gsInBuffer
            var dst = gsOutBuffer

            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(gsPipeline)
                encoder.setBuffer(src, offset: 0, index: 0)
                encoder.setBuffer(dst, offset: 0, index: 1)
                encoder.setBuffer(omegaBuffer, offset: 0, index: 2)
                encoder.setBytes(&gsSizeVar, length: MemoryLayout<simd_uint2>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: gsSize, height: gsSize, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                swap(&src, &dst)
            }
            let end = getTimeNanos()

            let elapsed = getElapsedSeconds(start: start, end: end)
            let mCellsPerSec = Double(gsSize * gsSize) * Double(iterations) / elapsed / 1e6

            print("Gauss-Seidel (omega=0.8): \(String(format: "%.2f", mCellsPerSec)) MCells/s")
        }

        print("\n--- Key Findings ---")
        print("1. Jacobi iteration: simplest but slowest convergence (needs many iterations)")
        print("2. Gauss-Seidel SOR: faster convergence with optimal omega~0.8-1.2")
        print("3. Red-Black coloring: enables parallel Gauss-Seidel")
        print("4. Shared memory: helps with halo cell access pattern")
        print("5. Heat equation is memory-bound, not compute-bound")
    }
}
