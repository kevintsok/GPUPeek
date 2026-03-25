import Foundation
import Metal

// MARK: - Texture Performance Benchmark

let textureShaders = """
#include <metal_stdlib>
using namespace metal;

// Texture read - sequential
kernel void texture_read_seq(texture2d<float, access::read> tex [[texture(0)]],
                          device float* out [[buffer(0)]],
                          constant uint2& size [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    float4 val = tex.read(gid);
    out[gid.y * size.x + gid.x] = val.x + val.y + val.z + val.w;
}

// Texture write - sequential
kernel void texture_write_seq(texture2d<float, access::write> tex [[texture(0)]],
                           device const float* in [[buffer(0)]],
                           constant uint2& size [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint idx = gid.y * size.x + gid.x;
    tex.write(float4(in[idx], in[idx], in[idx], 1.0f), gid);
}

// Buffer read - baseline
kernel void buffer_read(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = in[id] * 1.001f;
}

// Texture with sampler
kernel void texture_sample(texture2d<float, access::sample> tex [[texture(0)]],
                         sampler s [[sampler(0)]],
                         device float* out [[buffer(0)]],
                         constant float2& size [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(size.x) || gid.y >= uint(size.y)) return;
    float2 coord = float2(gid) / size;
    float4 val = tex.sample(s, coord);
    out[gid.y * uint(size.x) + gid.x] = val.x;
}

// Float4 buffer read for comparison
kernel void float4_buffer_read(device const float4* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = in[id];
    out[id * 4] = val.x;
    out[id * 4 + 1] = val.y;
    out[id * 4 + 2] = val.z;
    out[id * 4 + 3] = val.w;
}
"""

public struct TextureBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Texture Performance Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: textureShaders, options: nil) else {
            print("Failed to compile texture shaders")
            return
        }

        let texWidth = 1024
        let texHeight = 1024
        let textureSize = texWidth * texHeight

        // Create texture
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: texWidth,
            height: texHeight,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else {
            print("Failed to create texture")
            return
        }

        // Initialize texture
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

        // Create sampler
        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        guard let sampler = device.makeSampler(descriptor: samplerDesc) else {
            print("Failed to create sampler")
            return
        }

        // Create buffers
        guard let bufferIn = device.makeBuffer(length: textureSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: textureSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(textureSize)
        var texSize = SIMD2<UInt32>(UInt32(texWidth), UInt32(texHeight))

        let iterations = 50

        print("\n--- Texture vs Buffer Performance ---")

        // Test 1: Texture read
        if let texReadFunc = library.makeFunction(name: "texture_read_seq"),
           let texReadPipeline = try? device.makeComputePipelineState(function: texReadFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(texReadPipeline)
                encoder.setTexture(texture, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&texSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: texWidth, height: texHeight, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(textureSize) * 4.0 * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Texture Read: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Test 2: Buffer read baseline
        if let bufReadFunc = library.makeFunction(name: "buffer_read"),
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
            print("Buffer Read: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Test 3: Float4 buffer read
        if let float4Func = library.makeFunction(name: "float4_buffer_read"),
           let float4Pipeline = try? device.makeComputePipelineState(function: float4Func) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(float4Pipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: textureSize / 4, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(textureSize) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Float4 Buffer: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        print("\n--- Key Insights ---")
        print("1. Texture read involves sampler hardware - different path than buffer")
        print("2. Direct texture read (no filtering) still goes through sampler unit")
        print("3. Buffer access is direct to memory, no interpolation overhead")
        print("4. Textures are cached, so repeated reads may be faster")
    }
}
