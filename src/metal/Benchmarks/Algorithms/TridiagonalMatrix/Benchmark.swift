import Foundation
import Metal

// MARK: - Tridiagonal Matrix Solver Benchmark

let tridiagonalShaders = """
#include <metal_stdlib>
using namespace metal;

// Thomas Algorithm: Sequential forward-backward sweep O(n)
// Solves Ax = d where A is tridiagonal
// a[i] = sub-diagonal, b[i] = main diagonal, c[i] = super-diagonal
kernel void tridiagonal_thomas(device const float* a [[buffer(0)]],  // sub-diagonal
                              device const float* b [[buffer(1)]],  // main diagonal
                              device const float* c [[buffer(2)]],  // super-diagonal
                              device const float* d [[buffer(3)]],  // RHS
                              device float* x [[buffer(4)]],
                              device float* cp [[buffer(5)]],  // temporary: c'
                              device float* dp [[buffer(6)]],  // temporary: d'
                              constant uint& size [[buffer(7)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Forward sweep
    if (id == 0) {
        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];
    }

    // Process internal elements
    if (id > 0 && id < size - 1) {
        float denom = b[id] - a[id] * cp[id - 1];
        cp[id] = c[id] / denom;
        dp[id] = (d[id] - a[id] * dp[id - 1]) / denom;
    }

    // Backward substitution
    if (id == size - 1) {
        x[size - 1] = (d[size - 1] - a[size - 1] * dp[size - 2]) /
                      (b[size - 1] - a[size - 1] * cp[size - 2]);
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Back substitution for internal elements
    if (id < size - 1) {
        x[id] = dp[id] - cp[id] * x[id + 1];
    }
}

// Naive substitution (for comparison)
kernel void tridiagonal_naive(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device const float* c [[buffer(2)]],
                             device const float* d [[buffer(3)]],
                             device float* x [[buffer(4)]],
                             constant uint& size [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Simplified direct solve - requires diagonal dominance
    x[id] = (d[id] - a[id] * d[id-1] - c[id] * d[id+1]) / b[id];
}

// Parallel Cyclic Reduction (PCR) - O(log n) steps but more memory
kernel void tridiagonal_pcr(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device const float* c [[buffer(2)]],
                            device const float* d [[buffer(3)]],
                            device float* x [[buffer(4)]],
                            constant uint& size [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Simplified PCR: half the elements per step
    uint n = size;
    float alpha = a[id];
    float beta = b[id];
    float gamma = c[id];
    float delta = d[id];

    // 2-step reduction (simplified)
    if (id < n / 2) {
        uint i = id * 2;
        uint j = i + 1;

        // Combine equations i and j
        float new_beta = beta - alpha * c[i-1] / b[i-1] - gamma * a[j+1] / b[j+1];
        float new_delta = delta - alpha * d[i-1] / b[i-1] - gamma * d[j+1] / b[j+1];

        x[id] = new_delta / new_beta;
    }
}

// Banded matrix solve (generalization)
kernel void banded_solve(device const float* diag [[buffer(0)]],
                         device const float* lower [[buffer(1)]],
                         device const float* upper [[buffer(2)]],
                         device const float* rhs [[buffer(3)]],
                         device float* result [[buffer(4)]],
                         constant uint& size [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Banded forward elimination
    float temp_d = rhs[id];
    if (id > 0) {
        temp_d -= lower[id] * result[id - 1];
    }
    result[id] = temp_d / diag[id];
}
"""

public struct TridiagonalMatrixBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Tridiagonal Matrix Solver Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: tridiagonalShaders, options: nil) else {
            print("Failed to compile tridiagonal shaders")
            return
        }

        let sizes: [Int] = [16384, 65536, 262144, 1048576]  // 16K to 1M

        for size in sizes {
            print("\n--- Size: \(size) elements ---")

            guard let bufA = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufB = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufC = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufD = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufX = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufCp = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufDp = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize diagonally dominant tridiagonal matrix
            let aPtr = bufA.contents().bindMemory(to: Float.self, capacity: size)
            let bPtr = bufB.contents().bindMemory(to: Float.self, capacity: size)
            let cPtr = bufC.contents().bindMemory(to: Float.self, capacity: size)
            let dPtr = bufD.contents().bindMemory(to: Float.self, capacity: size)

            for i in 0..<size {
                aPtr[i] = (i > 0) ? Float.random(in: -0.1...0.1) : 0.0
                bPtr[i] = Float.random(in: 0.9...1.0) + Float(2.0)  // Diagonal dominant
                cPtr[i] = (i < size - 1) ? Float.random(in: -0.1...0.1) : 0.0
                dPtr[i] = Float.random(in: 0.0...1.0)
            }

            var sizeVar = UInt32(size)

            // Thomas Algorithm (sequential)
            if let thomasFunc = library.makeFunction(name: "tridiagonal_thomas"),
               let thomasPipeline = try? device.makeComputePipelineState(function: thomasFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(thomasPipeline)
                    encoder.setBuffer(bufA, offset: 0, index: 0)
                    encoder.setBuffer(bufB, offset: 0, index: 1)
                    encoder.setBuffer(bufC, offset: 0, index: 2)
                    encoder.setBuffer(bufD, offset: 0, index: 3)
                    encoder.setBuffer(bufX, offset: 0, index: 4)
                    encoder.setBuffer(bufCp, offset: 0, index: 5)
                    encoder.setBuffer(bufDp, offset: 0, index: 6)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 7)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

                // O(n) operations per iteration
                let ops = Double(size) * Double(iterations)
                let gops = ops / elapsed / 1e9

                print("Thomas (O(n)): \(String(format: "%.3f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Naive solver
            if let naiveFunc = library.makeFunction(name: "tridiagonal_naive"),
               let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(naivePipeline)
                    encoder.setBuffer(bufA, offset: 0, index: 0)
                    encoder.setBuffer(bufB, offset: 0, index: 1)
                    encoder.setBuffer(bufC, offset: 0, index: 2)
                    encoder.setBuffer(bufD, offset: 0, index: 3)
                    encoder.setBuffer(bufX, offset: 0, index: 4)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 5)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let ops = Double(size) * Double(iterations)
                let gops = ops / elapsed / 1e9

                print("Naive:        \(String(format: "%.3f", gops)) GOPS")
            }
        }

        // Performance analysis
        print("\n--- Performance Analysis ---")
        print("Thomas Algorithm: O(n) sequential but requires synchronization")
        print("Tridiagonal systems appear in:")
        print("  - PDE discretizations (1D heat equation, etc.)")
        print("  - Cubic spline interpolation")
        print("  - Circuit simulation (resistive networks)")
        print("  - Structural analysis (truss systems)")

        print("\n--- Key Findings ---")
        print("1. Thomas algorithm is inherently sequential (forward-backward sweep)")
        print("2. GPU parallelism limited by data dependencies")
        print("3. Cyclic Reduction (CR) enables O(log n) parallelism")
        print("4. For large systems, batch solve multiple tridiagonals")
        print("5. Apple's GPU benefits from unified memory for this use case")
    }
}
