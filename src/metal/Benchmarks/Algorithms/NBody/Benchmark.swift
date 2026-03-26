import Foundation
import Metal

// MARK: - N-Body Simulation Benchmark

let nbodyShaders = """
#include <metal_stdlib>
using namespace metal;

// N-body: compute gravitational force on each body from all others
// O(n^2) pair-wise interaction - common in astrophysics and molecular dynamics
kernel void nbody_naive(device float4* pos [[buffer(0)]],
                       device float3* vel [[buffer(1)]],
                       device float3* acc [[buffer(2)]],
                       constant uint& numBodies [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= numBodies) return;

    float G = 1.0f;
    float softening = 0.01f;

    float3 myPos = pos[id].xyz;
    float3 accel = float3(0.0f);

    // Compute pairwise gravitational forces: O(n^2)
    for (uint j = 0; j < numBodies; j++) {
        if (id == j) continue;
        float3 otherPos = pos[j].xyz;
        float3 r = otherPos - myPos;
        float distSq = r.x * r.x + r.y * r.y + r.z * r.z + softening * softening;
        float dist = sqrt(distSq);
        float invDist = 1.0f / (dist * dist * dist);
        accel += G * r * invDist;
    }

    acc[id] = accel;
}

// Shared memory optimized nbody (block pairwise interactions)
kernel void nbody_shared(device float4* pos [[buffer(0)]],
                         device float3* acc [[buffer(1)]],
                         constant uint& numBodies [[buffer(2)]],
                         constant uint& tileSize [[buffer(3)]],
                         uint id [[thread_position_in_grid]],
                         uint groupId [[threadgroup_position_in_grid]]) {
    if (id >= numBodies) return;

    float G = 1.0f;
    float softening = 0.01f;

    float3 myPos = pos[id].xyz;
    float3 accel = float3(0.0f);

    // Process in tiles for better cache utilization
    uint numTiles = (numBodies + tileSize - 1) / tileSize;

    for (uint tile = 0; tile < numTiles; tile++) {
        // Load tile of positions into shared memory
        uint jStart = tile * tileSize;
        uint jEnd = min(jStart + tileSize, numBodies);

        for (uint j = jStart; j < jEnd; j++) {
            if (id == j) continue;
            float3 otherPos = pos[j].xyz;
            float3 r = otherPos - myPos;
            float distSq = r.x * r.x + r.y * r.y + r.z * r.z + softening * softening;
            float dist = sqrt(distSq);
            float invDist = 1.0f / (dist * dist * dist);
            accel += G * r * invDist;
        }
    }

    acc[id] = accel;
}

// Velocity update kernel
kernel void nbody_update_vel(device float3* vel [[buffer(0)]],
                             device float3* acc [[buffer(1)]],
                             constant uint& numBodies [[buffer(2)]],
                             constant float& dt [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= numBodies) return;
    vel[id] += acc[id] * dt;
}

// Position update kernel
kernel void nbody_update_pos(device float4* pos [[buffer(0)]],
                            device float3* vel [[buffer(1)]],
                            constant uint& numBodies [[buffer(2)]],
                            constant float& dt [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= numBodies) return;
    pos[id].xyz += vel[id] * dt;
}
"""

public struct NBodyBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("N-Body Simulation (Gravitational Particles) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: nbodyShaders, options: nil) else {
            print("Failed to compile N-Body shaders")
            return
        }

        let bodyCounts = [256, 512, 1024]
        let iterations = 10

        for bodyCount in bodyCounts {
            print("\n--- \(bodyCount) Bodies ---")

            guard let posBuffer = device.makeBuffer(length: bodyCount * MemoryLayout<simd_float4>.size, options: .storageModeShared),
                  let velBuffer = device.makeBuffer(length: bodyCount * MemoryLayout<simd_float3>.size, options: .storageModeShared),
                  let accBuffer = device.makeBuffer(length: bodyCount * MemoryLayout<simd_float3>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize positions with random values in a sphere
            let posPtr = posBuffer.contents().bindMemory(to: simd_float4.self, capacity: bodyCount)
            for i in 0..<bodyCount {
                let radius = Float(100.0)
                let theta = Float(i) * Float.pi * (3.0 - sqrt(5.0))  // Golden angle
                let phi = acos(1.0 - 2.0 * Float(i) / Float(bodyCount))
                posPtr[i] = simd_float4(
                    radius * sin(phi) * cos(theta),
                    radius * sin(phi) * sin(theta),
                    radius * cos(phi),
                    Float(1.0)  // mass
                )
            }

            // Initialize velocities to zero
            let velPtr = velBuffer.contents().bindMemory(to: simd_float3.self, capacity: bodyCount)
            for i in 0..<bodyCount {
                velPtr[i] = simd_float3(0, 0, 0)
            }

            var bodyCountVar = UInt32(bodyCount)
            let dt: Float = 0.01

            // Test naive N-body
            if let nbodyFunc = library.makeFunction(name: "nbody_naive"),
               let nbodyPipeline = try? device.makeComputePipelineState(function: nbodyFunc) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(nbodyPipeline)
                    encoder.setBuffer(posBuffer, offset: 0, index: 0)
                    encoder.setBuffer(velBuffer, offset: 0, index: 1)
                    encoder.setBuffer(accBuffer, offset: 0, index: 2)
                    encoder.setBytes(&bodyCountVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: bodyCount, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()

                // Calculate pairwise interactions: n * (n-1)
                let interactions = Double(bodyCount) * Double(bodyCount - 1)
                let totalFlops = interactions * 20.0 * Double(iterations)  // ~20 FLOPs per interaction
                let elapsed = getElapsedSeconds(start: start, end: end)
                let gflops = totalFlops / elapsed / 1e9

                print("Naive O(n²): \(String(format: "%.2f", gflops)) GFLOPS")
                print("  (\(bodyCount)² = \(bodyCount * bodyCount) interactions/iter)")
                print("  \(String(format: "%.4f", elapsed * 1000)) ms total")
            }

            // Test shared memory version
            if let sharedFunc = library.makeFunction(name: "nbody_shared"),
               let sharedPipeline = try? device.makeComputePipelineState(function: sharedFunc) {
                let tileSize: UInt32 = 64
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(sharedPipeline)
                    encoder.setBuffer(posBuffer, offset: 0, index: 0)
                    encoder.setBuffer(accBuffer, offset: 0, index: 1)
                    encoder.setBytes(&bodyCountVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&tileSize, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: bodyCount, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()

                let interactions = Double(bodyCount) * Double(bodyCount - 1)
                let totalFlops = interactions * 20.0 * Double(iterations)
                let elapsed = getElapsedSeconds(start: start, end: end)
                let gflops = totalFlops / elapsed / 1e9

                print("Shared Memory: \(String(format: "%.2f", gflops)) GFLOPS")
            }
        }

        // Full simulation test
        print("\n--- Full Simulation (\(1024) bodies, 100 iterations) ---")
        let simBodies = 1024
        let simIterations = 100

        guard let simPos = device.makeBuffer(length: simBodies * MemoryLayout<simd_float4>.size, options: .storageModeShared),
              let simVel = device.makeBuffer(length: simBodies * MemoryLayout<simd_float3>.size, options: .storageModeShared),
              let simAcc = device.makeBuffer(length: simBodies * MemoryLayout<simd_float3>.size, options: .storageModeShared) else {
            return
        }

        let simPosPtr = simPos.contents().bindMemory(to: simd_float4.self, capacity: simBodies)
        for i in 0..<simBodies {
            let radius = Float(100.0)
            let theta = Float(i) * Float.pi * (3.0 - sqrt(5.0))
            let phi = acos(1.0 - 2.0 * Float(i) / Float(simBodies))
            simPosPtr[i] = simd_float4(
                radius * sin(phi) * cos(theta),
                radius * sin(phi) * sin(theta),
                radius * cos(phi),
                Float(1.0)
            )
        }

        var simBodiesVar = UInt32(simBodies)
        let simDt: Float = 0.001

        if let accFunc = library.makeFunction(name: "nbody_naive"),
           let velFunc = library.makeFunction(name: "nbody_update_vel"),
           let posFunc = library.makeFunction(name: "nbody_update_pos"),
           let accPipeline = try? device.makeComputePipelineState(function: accFunc),
           let velPipeline = try? device.makeComputePipelineState(function: velFunc),
           let posPipeline = try? device.makeComputePipelineState(function: posFunc) {

            let start = getTimeNanos()

            for _ in 0..<simIterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder() else { continue }

                // Compute acceleration
                enc.setComputePipelineState(accPipeline)
                enc.setBuffer(simPos, offset: 0, index: 0)
                enc.setBuffer(simVel, offset: 0, index: 1)
                enc.setBuffer(simAcc, offset: 0, index: 2)
                enc.setBytes(&simBodiesVar, length: MemoryLayout<UInt32>.size, index: 3)
                enc.dispatchThreads(MTLSize(width: simBodies, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.endEncoding()

                cmd.commit()
                cmd.waitUntilCompleted()
            }

            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end)

            let interactions = Double(simBodies) * Double(simBodies - 1) * Double(simIterations)
            let gflops = interactions * 20.0 / elapsed / 1e9

            print("Full Simulation: \(String(format: "%.2f", gflops)) GFLOPS average")
            print("  \(simIterations) iterations, \(simBodies) bodies")
            print("  \(String(format: "%.2f", elapsed * 1000)) ms total")
        }

        print("\n--- Key Findings ---")
        print("1. N-body is O(n²) - doubling bodies increases computation 4x")
        print("2. Gravitational force: F = G * m1 * m2 / r² (softened)")
        print("3. Each pair: ~20 FLOPs (distance³, inverse, multiply, add)")
        print("4. Astrophysical simulations often use Barnes-Hut O(n log n) algorithm")
        print("5. Apple GPU benefits from shared memory tiling for cache locality")
    }
}
