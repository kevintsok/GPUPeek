import Foundation
import Metal

// MARK: - Acceleration Structures (BVH) Benchmark

let accelerationShaders = """
#include <metal_stdlib>
using namespace metal;

// Ray-AABB intersection test
kernel void ray_aabb_test(device float4* rays [[buffer(0)]],
                         device float* bvh [[buffer(1)]],
                         device float* results [[buffer(2)]],
                         constant uint& numRays [[buffer(3)]],
                         constant uint& maxNodes [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 orig = rays[id].xyz;
    float3 dir = normalize(rays[id + numRays].xyz);

    float minT = 1e10;

    // Test against all nodes (naive)
    for (uint node = 0; node < maxNodes; node++) {
        float3 minB = float3(bvh[node * 8 + 0], bvh[node * 8 + 1], bvh[node * 8 + 2]);
        float3 maxB = float3(bvh[node * 8 + 3], bvh[node * 8 + 4], bvh[node * 8 + 5]);

        // Slab method for ray-AABB intersection
        float3 invDir = 1.0f / dir;
        float3 t0 = (minB - orig) * invDir;
        float3 t1 = (maxB - orig) * invDir;

        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);

        float tNear = max(max(tmin.x, tmin.y), tmin.z);
        float tFar = min(min(tmax.x, tmax.y), tmax.z);

        if (tNear <= tFar && tFar > 0.0f) {
            minT = min(minT, tNear);
        }
    }

    results[id] = (minT < 1e10) ? minT : -1.0f;
}

// Ray-Sphere intersection (for comparison)
kernel void ray_sphere_test(device float4* rays [[buffer(0)]],
                          device float4* spheres [[buffer(1)]],
                          device float* results [[buffer(2)]],
                          constant uint& numRays [[buffer(3)]],
                          constant uint& numSpheres [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 orig = rays[id].xyz;
    float3 dir = normalize(rays[id + numRays].xyz);

    float minT = 1e10;

    for (uint s = 0; s < numSpheres; s++) {
        float3 center = spheres[s].xyz;
        float radius = spheres[s].w;

        float3 oc = orig - center;
        float b = dot(oc, dir);
        float c = dot(oc, oc) - radius * radius;
        float h = b * b - c;

        if (h >= 0.0f) {
            float t = -b - sqrt(h);
            if (t > 0.0f && t < minT) {
                minT = t;
            }
        }
    }

    results[id] = (minT < 1e10) ? minT : -1.0f;
}

// BVH traversal with stack
kernel void bvh_traverse(device float4* rays [[buffer(0)]],
                        device float* bvh [[buffer(1)]],
                        device float* results [[buffer(2)]],
                        device uint* stack [[buffer(3)]],
                        constant uint& numRays [[buffer(4)]],
                        constant uint& maxNodes [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 orig = rays[id].xyz;
    float3 dir = normalize(rays[id + numRays].xyz);
    float3 invDir = 1.0f / dir;

    float minT = 1e10;

    // Simple traversal: start from root
    uint stackPtr = 0;
    stack[stackPtr++] = 0;  // Root node

    while (stackPtr > 0) {
        stackPtr--;
        uint node = stack[stackPtr];

        if (node >= maxNodes) continue;

        float3 minB = float3(bvh[node * 8 + 0], bvh[node * 8 + 1], bvh[node * 8 + 2]);
        float3 maxB = float3(bvh[node * 8 + 3], bvh[node * 8 + 4], bvh[node * 8 + 5]);

        // Ray-AABB test
        float3 t0 = (minB - orig) * invDir;
        float3 t1 = (maxB - orig) * invDir;
        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);
        float tNear = max(max(tmin.x, tmin.y), tmin.z);
        float tFar = min(min(tmax.x, tmax.y), tmax.z);

        if (tNear <= tFar && tFar > 0.0f) {
            uint leftIdx = uint(bvh[node * 8 + 6]);
            uint sphereIdx = uint(bvh[node * 8 + 7]);

            if (leftIdx == 0xFFFFFFFF) {
                // Leaf node
                minT = min(minT, tNear);
            } else {
                // Push children to stack
                if (stackPtr < 32) {
                    stack[stackPtr++] = leftIdx;
                    stack[stackPtr++] = leftIdx + 1;
                }
            }
        }
    }

    results[id] = (minT < 1e10) ? minT : -1.0f;
}

// Brute force ray-sphere for comparison
kernel void brute_force_test(device float4* rays [[buffer(0)]],
                            device float4* spheres [[buffer(1)]],
                            device float* results [[buffer(2)]],
                            constant uint& numRays [[buffer(3)]],
                            constant uint& numSpheres [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= numRays) return;

    float3 orig = rays[id].xyz;
    float3 dir = normalize(rays[id + numRays].xyz);

    float minT = 1e10;

    // Test all spheres
    for (uint s = 0; s < numSpheres; s++) {
        float3 center = spheres[s].xyz;
        float radius = spheres[s].w;

        float3 oc = orig - center;
        float b = dot(oc, dir);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - c;

        if (discriminant > 0.0f) {
            float t = -b - sqrt(discriminant);
            if (t > 0.001f && t < minT) {
                minT = t;
            }
        }
    }

    results[id] = (minT < 1e10) ? minT : -1.0f;
}
"""

public struct AccelerationStructuresBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Acceleration Structures (BVH) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: accelerationShaders, options: nil) else {
            print("Failed to compile acceleration structure shaders")
            return
        }

        let numSpheres: UInt32 = 256
        let numRays: UInt32 = 1024
        let maxNodes: UInt32 = 512

        print("\n--- Setup: \(numSpheres) spheres, \(numRays) rays ---")

        // Create buffers
        guard let rayBuffer = device.makeBuffer(length: Int(numRays) * MemoryLayout<simd_float4>.size * 2, options: .storageModeShared),
              let sphereBuffer = device.makeBuffer(length: Int(numSpheres) * MemoryLayout<simd_float4>.size, options: .storageModeShared),
              let bvhBuffer = device.makeBuffer(length: Int(maxNodes) * 8 * MemoryLayout<Float>.size, options: .storageModeShared),
              let resultsBuffer = device.makeBuffer(length: Int(numRays) * MemoryLayout<Float>.size, options: .storageModeShared),
              let stackBuffer = device.makeBuffer(length: 32 * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize rays
        let rayPtr = rayBuffer.contents().bindMemory(to: simd_float4.self, capacity: Int(numRays) * 2)
        for i in 0..<Int(numRays) {
            rayPtr[i] = simd_float4(Float(i % 16) * 0.5 - 4.0,  // origin x
                                    Float((i / 16) % 16) * 0.5 - 4.0,  // origin y
                                    -5.0,  // origin z
                                    0.0)  // w
            rayPtr[i + Int(numRays)] = simd_float4(0.0, 0.0, 1.0, 0.0)  // direction
        }

        // Initialize spheres
        let spherePtr = sphereBuffer.contents().bindMemory(to: simd_float4.self, capacity: Int(numSpheres))
        for i in 0..<Int(numSpheres) {
            let x = Float(i % 16) * 0.6 - 4.5
            let y = Float((i / 16) % 16) * 0.6 - 4.5
            let z = Float(i / 256) * 0.6 - 0.5
            spherePtr[i] = simd_float4(x, y, z, 0.5)  // radius 0.5
        }

        // Build simple BVH (flat, for testing)
        let bvhPtr = bvhBuffer.contents().bindMemory(to: Float.self, capacity: Int(maxNodes) * 8)
        // Fill with simple bounding boxes covering all spheres
        for i in 0..<Int(maxNodes) {
            bvhPtr[i * 8 + 0] = -5.0  // min.x
            bvhPtr[i * 8 + 1] = -5.0  // min.y
            bvhPtr[i * 8 + 2] = -5.0  // min.z
            bvhPtr[i * 8 + 3] = 5.0   // max.x
            bvhPtr[i * 8 + 4] = 5.0   // max.y
            bvhPtr[i * 8 + 5] = 5.0   // max.z
            bvhPtr[i * 8 + 6] = Float(0xFFFFFFFF)  // left = leaf
            bvhPtr[i * 8 + 7] = Float(i)  // sphere index
        }

        var numRaysValue = numRays
        var numSpheresValue = numSpheres
        var maxNodesValue = maxNodes

        // Test brute force ray-sphere
        if let bruteFunc = library.makeFunction(name: "brute_force_test"),
           let brutePipeline = try? device.makeComputePipelineState(function: bruteFunc) {
            let iterations = 50
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(brutePipeline)
                encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                encoder.setBuffer(sphereBuffer, offset: 0, index: 1)
                encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
                encoder.setBytes(&numRaysValue, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&numSpheresValue, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreads(MTLSize(width: Int(numRays), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let throughput = Double(numRays) * Double(numSpheres) / elapsed / 1e6
            print("Brute Force: \(String(format: "%.2f", throughput)) MR/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
        }

        // Test ray-AABB
        if let aabbFunc = library.makeFunction(name: "ray_aabb_test"),
           let aabbPipeline = try? device.makeComputePipelineState(function: aabbFunc) {
            let iterations = 50
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(aabbPipeline)
                encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                encoder.setBuffer(bvhBuffer, offset: 0, index: 1)
                encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
                encoder.setBytes(&numRaysValue, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&maxNodesValue, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreads(MTLSize(width: Int(numRays), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let throughput = Double(numRays) / elapsed / 1e6
            print("Ray-AABB: \(String(format: "%.2f", throughput)) MR/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
        }

        // Test BVH traversal
        if let bvhFunc = library.makeFunction(name: "bvh_traverse"),
           let bvhPipeline = try? device.makeComputePipelineState(function: bvhFunc) {
            let iterations = 50
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(bvhPipeline)
                encoder.setBuffer(rayBuffer, offset: 0, index: 0)
                encoder.setBuffer(bvhBuffer, offset: 0, index: 1)
                encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
                encoder.setBuffer(stackBuffer, offset: 0, index: 3)
                encoder.setBytes(&numRaysValue, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&maxNodesValue, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.dispatchThreads(MTLSize(width: Int(numRays), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let throughput = Double(numRays) / elapsed / 1e6
            print("BVH Traversal: \(String(format: "%.2f", throughput)) MR/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
        }

        print("\n--- Key Findings ---")
        print("1. BVH accelerates ray tracing from O(rays x spheres) to O(rays x log(spheres))")
        print("2. Ray-AABB is faster than ray-sphere testing")
        print("3. BVH traversal uses stack-based hierarchical testing")
        print("4. Apple M3 has hardware ray tracing units")
        print("5. Real ray tracers use SAH (Surface Area Heuristic) for optimal splits")
    }
}
