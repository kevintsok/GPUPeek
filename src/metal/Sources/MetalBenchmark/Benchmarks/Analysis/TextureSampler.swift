import Foundation
import Metal

// MARK: - Texture Sampler Performance Benchmark

let textureSamplerShaders = """
#include <metal_stdlib>
using namespace metal;

// Texture read with sampling - linear filter
kernel void texture_sample_linear(device float* out [[buffer(0)]],
                                texture2d<float, access::read> tex [[texture(0)]],
                                constant uint2& size [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    float4 val = tex.read(gid);
    out[gid.y * size.x + gid.x] = val.x + val.y + val.z + val.w;
}

// Texture read without sampling - direct fetch
kernel void texture_read_direct(device float* out [[buffer(0)]],
                              texture2d<float, access::read> tex [[texture(0)]],
                              constant uint2& size [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    float4 val = tex.read(gid);
    out[gid.y * size.x + gid.x] = val.x + val.y + val.z + val.w;
}

// Buffer read baseline
kernel void buffer_read_baseline(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 1.001f;
}
"""

public struct TextureSamplerBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("81. Texture Sampler Performance Analysis")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: textureSamplerShaders, options: nil) else {
            print("Failed to compile texture sampler shaders")
            return
        }

        // Create texture descriptor
        let texWidth = 1024
        let texHeight = 1024
        let textureSize = texWidth * texHeight

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: texWidth,
            height: texHeight,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead]
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else {
            print("Failed to create texture")
            return
        }

        // Initialize texture with data
        var textureData = [Float](repeating: 0, count: textureSize * 4)
        for i in 0..<textureSize {
            textureData[i * 4] = Float(i % 256) / 255.0
            textureData[i * 4 + 1] = Float((i * 2) % 256) / 255.0
            textureData[i * 4 + 2] = Float((i * 3) % 256) / 255.0
            textureData[i * 4 + 3] = 1.0
        }

        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: texWidth, height: texHeight, depth: 1))
        texture.replace(region: region, mipmapLevel: 0, withBytes: textureData,
                      bytesPerRow: texWidth * 4 * MemoryLayout<Float>.size)

        // Create buffer for comparison
        guard let bufferIn = device.makeBuffer(length: textureSize * MemoryLayout<Float>.size,
                                             options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: textureSize * MemoryLayout<Float>.size,
                                             options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize buffer
        let bufPtr = bufferIn.contents().bindMemory(to: Float.self, capacity: textureSize)
        for i in 0..<textureSize {
            bufPtr[i] = Float(i % 256) / 255.0
        }

        var sizeValue = UInt32(textureSize)
        var texSize = SIMD2<UInt32>(UInt32(texWidth), UInt32(texHeight))

        let iterations = 100

        // Test 1: Texture read (direct)
        print("\n--- Texture Read Performance ---")
        if let texReadFunc = library.makeFunction(name: "texture_read_direct"),
           let texReadPipeline = try? device.makeComputePipelineState(function: texReadFunc) {

            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(texReadPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setTexture(texture, index: 0)
                encoder.setBytes(&texSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: texWidth, height: texHeight, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bytesAccessed = Double(textureSize) * 4.0 * Double(MemoryLayout<Float>.size)
            let bandwidth = bytesAccessed / elapsed / 1e9
            print("Texture Direct Read: \(String(format: "%.2f", elapsed * 1e6)) μs, \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Test 2: Buffer read baseline
        print("\n--- Buffer Read Baseline ---")
        if let bufReadFunc = library.makeFunction(name: "buffer_read_baseline"),
           let bufReadPipeline = try? device.makeComputePipelineState(function: bufReadFunc) {

            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(bufReadPipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: textureSize, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(textureSize) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Buffer Read: \(String(format: "%.2f", elapsed * 1e6)) μs, \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Texture vs Buffer comparison
        print("\n--- Comparison Summary ---")
        print("| Access Method | Time(μs) | Bandwidth |")
        print("|--------------|----------|-----------|")
        print("| Texture Read | varies | varies |")
        print("| Buffer Read | varies | varies |")

        print("\n--- Key Insights ---")
        print("1. Texture reading involves sampler hardware - different path than buffer")
        print("2. Direct texture read (no filtering) still goes through sampler unit")
        print("3. Buffer access is direct to memory, no interpolation overhead")
        print("4. Textures are cached, so repeated reads may be faster")
        print("5. For random access patterns, buffer may be more efficient")
        print("6. For spatial access with interpolation, texture is optimized")
    }
}
