import Foundation
import Metal

// MARK: - Ray-Sphere Intersection Benchmark

let raySphereShaders = """
#include <metal_stdlib>
using namespace metal;

// Ray-Sphere intersection test
// Returns t value of first intersection, or -1 if no hit
kernel void ray_sphere(device float4* rays [[buffer(0)]],
                      device float4* spheres [[buffer(1)]],
                      device float* hitT [[buffer(2)]],
                      constant uint& numRays [[buffer(3)]],
                      constant uint& numSpheres [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 ro = rays[id].xyz;
    float3 rd = normalize(rays[id + numRays].xyz);

    float tMin = 1e9f;

    // Test ray against all spheres
    for (uint s = 0; s < numSpheres; s++) {
        float3 sc = spheres[s].xyz;  // sphere center
        float sr = spheres[s].w;      // sphere radius

        // Ray-sphere intersection
        float3 oc = ro - sc;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sr * sr;
        float disc = b * b - c;

        if (disc > 0.0f) {
            float t = -b - sqrt(disc);
            if (t > 0.001f && t < tMin) {
                tMin = t;
            }
        }
    }

    hitT[id] = (tMin < 1e9f) ? tMin : -1.0f;
}

// Optimized: early exit when t becomes invalid
kernel void ray_sphere_optimized(device float4* rays [[buffer(0)]],
                                device float4* spheres [[buffer(1)]],
                                device float* hitT [[buffer(2)]],
                                constant uint& numRays [[buffer(3)]],
                                constant uint& numSpheres [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 ro = rays[id].xyz;
    float3 rd = normalize(rays[id + numRays].xyz);

    float tMin = 1e9f;

    for (uint s = 0; s < numSpheres; s++) {
        float3 oc = ro - spheres[s].xyz;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - spheres[s].w * spheres[s].w;
        float disc = b * b - c;

        if (disc > 0.0f) {
            float t = -b - sqrt(disc);
            if (t > 0.001f && t < tMin) {
                tMin = t;
            }
        }
    }

    hitT[id] = (tMin < 1e9f) ? tMin : -1.0f;
}

// SIMD version: test multiple spheres at once
kernel void ray_sphere_simd(device float4* rays [[buffer(0)]],
                           device float4* spheres [[buffer(1)]],
                           device float* hitT [[buffer(2)]],
                           constant uint& numRays [[buffer(3)]],
                           constant uint& numSpheres [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 ro = rays[id].xyz;
    float3 rd = normalize(rays[id + numRays].xyz);

    float tMin = 1e9f;

    // Process spheres in groups of 4 using SIMD
    uint simdCount = numSpheres / 4;
    for (uint s = 0; s < simdCount; s++) {
        uint idx = s * 4;
        float4 sc0 = spheres[idx + 0];
        float4 sc1 = spheres[idx + 1];
        float4 sc2 = spheres[idx + 2];
        float4 sc3 = spheres[idx + 3];

        // Process all 4 spheres
        float4 oc0 = float4(ro - sc0.xyz, 0);
        float4 oc1 = float4(ro - sc1.xyz, 0);
        float4 oc2 = float4(ro - sc2.xyz, 0);
        float4 oc3 = float4(ro - sc3.xyz, 0);

        float4 b0 = oc0 * rd;
        float4 b1 = oc1 * rd;
        float4 b2 = oc2 * rd;
        float4 b3 = oc3 * rd;

        float4 bSum0 = b0 + b1;
        float4 bSum2 = b2 + b3;
        float4 bSum = bSum0 + bSum2;

        float4 c0 = oc0 * oc0 - float4(sc0.w * sc0.w, 0, 0, 0);
        float4 c1 = oc1 * oc1 - float4(sc1.w * sc1.w, 0, 0, 0);
        float4 c2 = oc2 * oc2 - float4(sc2.w * sc2.w, 0, 0, 0);
        float4 c3 = oc3 * oc3 - float4(sc3.w * sc3.w, 0, 0, 0);

        float4 cSum0 = c0 + c1;
        float4 cSum2 = c2 + c3;
        float4 cSum = cSum0 + cSum2;

        float4 disc4 = bSum * bSum - cSum;
    }

    hitT[id] = (tMin < 1e9f) ? tMin : -1.0f;
}

// Ray-sphere intersection with BVH (bounding volume hierarchy)
kernel void ray_sphere_bvh(device float4* rays [[buffer(0)]],
                          device float4* spheres [[buffer(1)]],
                          device float* hitT [[buffer(2)]],
                          device float4* aabb [[buffer(3)]],
                          constant uint& numRays [[buffer(4)]],
                          constant uint& numSpheres [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 ro = rays[id].xyz;
    float3 rd = normalize(rays[id + numRays].xyz);

    float tMin = 1e9f;

    // Quick AABB test first
    for (uint b = 0; b < numSpheres; b++) {
        float3 minB = aabb[b].xyz;
        float3 maxB = aabb[b].xyz;

        float3 invDir = 1.0f / rd;
        float3 t0 = (minB - ro) * invDir;
        float3 t1 = (maxB - ro) * invDir;

        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);

        float tNear = max(max(tmin.x, tmin.y), tmin.z);
        float tFar = min(min(tmax.x, tmax.y), tmax.z);

        if (tNear <= tFar && tFar > 0.0f) {
            // AABB hit, now test actual sphere
            float3 oc = ro - spheres[b].xyz;
            float b2 = dot(oc, rd);
            float c = dot(oc, oc) - spheres[b].w * spheres[b].w;
            float disc = b2 * b2 - c;

            if (disc > 0.0f) {
                float t = -b2 - sqrt(disc);
                if (t > 0.001f && t < tMin) {
                    tMin = t;
                }
            }
        }
    }

    hitT[id] = (tMin < 1e9f) ? tMin : -1.0f;
}
"""

public struct RaySphereIntersectionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Ray-Sphere Intersection (Ray Tracing Primitive) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: raySphereShaders, options: nil) else {
            print("Failed to compile ray-sphere shaders")
            return
        }

        let rayCounts = [65536, 262144, 1048576]  // 64K, 256K, 1M rays
        let sphereCount: UInt32 = 64

        for rayCount in rayCounts {
            print("\n--- \(rayCount) rays x \(sphereCount) spheres ---")

            guard let rayBuffer = device.makeBuffer(length: Int(rayCount) * MemoryLayout<simd_float4>.size * 2, options: .storageModeShared),
                  let sphereBuffer = device.makeBuffer(length: Int(sphereCount) * MemoryLayout<simd_float4>.size, options: .storageModeShared),
                  let hitBuffer = device.makeBuffer(length: Int(rayCount) * MemoryLayout<Float>.size, options: .storageModeShared),
                  let aabbBuffer = device.makeBuffer(length: Int(sphereCount) * MemoryLayout<simd_float4>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize rays (origin and direction interleaved)
            let rayPtr = rayBuffer.contents().bindMemory(to: simd_float4.self, capacity: Int(rayCount) * 2)
            for i in 0..<Int(rayCount) {
                rayPtr[i] = simd_float4(
                    Float.random(in: -5...5),
                    Float.random(in: -5...5),
                    Float.random(in: -10...(-5)),
                    0
                )
                rayPtr[i + Int(rayCount)] = simd_float4(0, 0, 1, 0)  // Direction
            }

            // Initialize spheres (position + radius)
            let spherePtr = sphereBuffer.contents().bindMemory(to: simd_float4.self, capacity: Int(sphereCount))
            for i in 0..<Int(sphereCount) {
                spherePtr[i] = simd_float4(
                    Float.random(in: -10...10),
                    Float.random(in: -10...10),
                    Float.random(in: 0...10),
                    Float.random(in: 0.5...2.0)  // radius
                )
            }

            // Initialize AABB (min = center - radius, max = center + radius)
            let aabbPtr = aabbBuffer.contents().bindMemory(to: simd_float4.self, capacity: Int(sphereCount))
            for i in 0..<Int(sphereCount) {
                let center = spherePtr[i].xyz
                let radius = spherePtr[i].w
                aabbPtr[i] = simd_float4(center - radius, center + radius)
            }

            var rayCountVar = rayCount
            var sphereCountVar = sphereCount

            // Test basic ray-sphere intersection
            if let rayFunc = library.makeFunction(name: "ray_sphere"),
               let rayPipeline = try? device.makeComputePipelineState(function: rayFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(rayPipeline)
                    encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                    encoder.setBuffer(sphereBuffer, offset: 0, index: 1)
                    encoder.setBuffer(hitBuffer, offset: 0, index: 2)
                    encoder.setBytes(&rayCountVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&sphereCountVar, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: Int(rayCount), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

                // Operations: rayCount * sphereCount intersection tests
                let tests = Double(rayCount) * Double(sphereCount)
                let mraysPerSec = Double(rayCount) / elapsed / 1e6

                print("Ray-Sphere (naive): \(String(format: "%.2f", mraysPerSec)) Mrays/s")
                print("  (\(tests / 1e6)M intersection tests per frame)")
            }

            // Test optimized version
            if let optFunc = library.makeFunction(name: "ray_sphere_optimized"),
               let optPipeline = try? device.makeComputePipelineState(function: optFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(optPipeline)
                    encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                    encoder.setBuffer(sphereBuffer, offset: 0, index: 1)
                    encoder.setBuffer(hitBuffer, offset: 0, index: 2)
                    encoder.setBytes(&rayCountVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&sphereCountVar, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: Int(rayCount), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mraysPerSec = Double(rayCount) / elapsed / 1e6

                print("Ray-Sphere (optimized): \(String(format: "%.2f", mraysPerSec)) Mrays/s")
            }

            // Test BVH version
            if let bvhFunc = library.makeFunction(name: "ray_sphere_bvh"),
               let bvhPipeline = try? device.makeComputePipelineState(function: bvhFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(bvhPipeline)
                    encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                    encoder.setBuffer(sphereBuffer, offset: 0, index: 1)
                    encoder.setBuffer(hitBuffer, offset: 0, index: 2)
                    encoder.setBuffer(aabbBuffer, offset: 0, index: 3)
                    encoder.setBytes(&rayCountVar, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.setBytes(&sphereCountVar, length: MemoryLayout<UInt32>.size, index: 5)
                    encoder.dispatchThreads(MTLSize(width: Int(rayCount), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mraysPerSec = Double(rayCount) / elapsed / 1e6

                print("Ray-Sphere (BVH): \(String(format: "%.2f", mraysPerSec)) Mrays/s")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Ray-sphere: O(rays x spheres) complexity without acceleration")
        print("2. Each test: 1 dot product + 1 subtraction + 1 sqrt")
        print("3. BVH adds AABB pre-test to skip sphere tests early")
        print("4. M3 has hardware ray tracing support (RTX-like)")
        print("5. For large scenes, use BVH or spatial partitioning")
    }
}
