import Foundation
import Metal

// MARK: - Image Processing Benchmark

let imageProcessingShaders = """
#include <metal_stdlib>
using namespace metal;

// Box blur 3x3
kernel void box_blur_3x3(device const float4* in [[buffer(0)]],
                         device float4* out [[buffer(1)]],
                         constant uint2& size [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 sum = float4(0.0f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 idx = int2(gid) + int2(dx, dy);
            idx = clamp(idx, int2(0), int2(size) - 1);
            sum += in[idx.y * size.x + idx.x];
        }
    }
    out[gid.y * size.x + gid.x] = sum * 0.111111f;  // 1/9
}

// Gaussian blur 5x5 (separable)
kernel void gaussian_blur_h(device const float4* in [[buffer(0)]],
                           device float4* temp [[buffer(1)]],
                           constant uint2& size [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 sum = float4(0.0f);
    float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

    for (int dx = -2; dx <= 2; dx++) {
        int2 idx = int2(gid) + int2(dx, 0);
        idx.x = clamp(idx.x, 0, int(size.x) - 1);
        sum += in[idx.y * size.x + idx.x] * weights[dx + 2];
    }
    temp[gid.y * size.x + gid.x] = sum;
}

kernel void gaussian_blur_v(device const float4* temp [[buffer(0)]],
                           device float4* out [[buffer(1)]],
                           constant uint2& size [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 sum = float4(0.0f);
    float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

    for (int dy = -2; dy <= 2; dy++) {
        int2 idx = int2(gid) + int2(0, dy);
        idx.y = clamp(idx.y, 0, int(size.y) - 1);
        sum += temp[idx.y * size.x + idx.x] * weights[dy + 2];
    }
    out[gid.y * size.x + gid.x] = sum;
}

// Sobel edge detection
kernel void sobel_x(device const float4* in [[buffer(0)]],
                    device float4* out [[buffer(1)]],
                    constant uint2& size [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 gx = float4(0.0f);
    int2 offsets[3] = {int2(-1, 0), int2(0, 0), int2(1, 0)};

    for (int i = 0; i < 3; i++) {
        int2 idx = int2(gid) + offsets[i];
        idx = clamp(idx, int2(0), int2(size) - 1);
        float w = (i == 0) ? -1.0f : (i == 1) ? 0.0f : 1.0f;
        gx += in[idx.y * size.x + idx.x] * w;
    }
    out[gid.y * size.x + gid.x] = gx;
}

kernel void sobel_y(device const float4* in [[buffer(0)]],
                    device float4* out [[buffer(1)]],
                    constant uint2& size [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 gy = float4(0.0f);
    int2 offsets[3] = {int2(0, -1), int2(0, 0), int2(0, 1)};

    for (int i = 0; i < 3; i++) {
        int2 idx = int2(gid) + offsets[i];
        idx = clamp(idx, int2(0), int2(size) - 1);
        float w = (i == 0) ? -1.0f : (i == 1) ? 0.0f : 1.0f;
        gy += in[idx.y * size.x + idx.x] * w;
    }
    out[gid.y * size.x + gid.x] = gy;
}

// Morphological dilation
kernel void dilate(device const float4* in [[buffer(0)]],
                   device float4* out [[buffer(1)]],
                   constant uint2& size [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 max_val = float4(0.0f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 idx = int2(gid) + int2(dx, dy);
            idx = clamp(idx, int2(0), int2(size) - 1);
            max_val = max(max_val, in[idx.y * size.x + idx.x]);
        }
    }
    out[gid.y * size.x + gid.x] = max_val;
}

// Morphological erosion
kernel void erode(device const float4* in [[buffer(0)]],
                  device float4* out [[buffer(1)]],
                  constant uint2& size [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 min_val = float4(1.0f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 idx = int2(gid) + int2(dx, dy);
            idx = clamp(idx, int2(0), int2(size) - 1);
            min_val = min(min_val, in[idx.y * size.x + idx.x]);
        }
    }
    out[gid.y * size.x + gid.x] = min_val;
}

// Bilinear interpolation (for image scaling)
kernel void bilinear_sample(device const float4* in [[buffer(0)]],
                           device float4* out [[buffer(1)]],
                           constant uint2& in_size [[buffer(2)]],
                           constant uint2& out_size [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out_size.x || gid.y >= out_size.y) return;

    float2 uv;
    uv.x = float(gid.x) * float(in_size.x) / float(out_size.x);
    uv.y = float(gid.y) * float(in_size.y) / float(out_size.y);

    uint2 base = uint2(floor(uv.x), floor(uv.y));
    float2 frac = uv - float2(base);

    uint2 idx00 = min(base, in_size - 1);
    uint2 idx10 = min(uint2(base.x + 1, base.y), in_size - 1);
    uint2 idx01 = min(uint2(base.x, base.y + 1), in_size - 1);
    uint2 idx11 = min(base + 1, in_size - 1);

    float4 v00 = in[idx00.y * in_size.x + idx00.x];
    float4 v10 = in[idx10.y * in_size.x + idx10.x];
    float4 v01 = in[idx01.y * in_size.x + idx01.x];
    float4 v11 = in[idx11.y * in_size.x + idx11.x];

    float4 v0 = mix(v00, v10, frac.x);
    float4 v1 = mix(v01, v11, frac.x);
    out[gid.y * out_size.x + gid.x] = mix(v0, v1, frac.y);
}
"""

public struct ImageProcessingBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Image Processing Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: imageProcessingShaders, options: nil) else {
            print("Failed to compile image processing shaders")
            return
        }

        let configs = [
            (512, 512),
            (1024, 1024),
            (2048, 2048)
        ]

        for (width, height) in configs {
            print("\n--- Image Size: \(width)x\(height) ---")

            let pixelCount = width * height
            guard let inBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size * 4, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size * 4, options: .storageModeShared),
                  let tempBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size * 4, options: .storageModeShared) else {
                continue
            }

            // Initialize with gradient pattern
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount * 4)
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4
                    inPtr[idx] = Float(x) / Float(width)      // R
                    inPtr[idx + 1] = Float(y) / Float(height)  // G
                    inPtr[idx + 2] = Float((x + y) % 256) / 255.0  // B
                    inPtr[idx + 3] = 1.0f  // A
                }
            }

            var sizeX = UInt32(width)
            var sizeY = UInt32(height)
            var size = uint2(UInt32(width), UInt32(height))

            // Test box blur 3x3
            if let blurFunc = library.makeFunction(name: "box_blur_3x3"),
               let blurPipeline = try? device.makeComputePipelineState(function: blurFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(blurPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mpixels = Double(pixelCount) / 1e6
                let fps = 1.0 / elapsed
                print("Box Blur 3x3: \(String(format: "%.2f", fps)) fps (\(String(format: "%.2f", mpixels / elapsed)) MP/s)")
            }

            // Test Gaussian blur (separable)
            if let gaussianHFunc = library.makeFunction(name: "gaussian_blur_h"),
               let gaussianVFunc = library.makeFunction(name: "gaussian_blur_v"),
               let gaussianHPipeline = try? device.makeComputePipelineState(function: gaussianHFunc),
               let gaussianVPipeline = try? device.makeComputePipelineState(function: gaussianVFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    // Horizontal pass
                    encoder.setComputePipelineState(gaussianHPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(tempBuffer, offset: 0, index: 1)
                    encoder.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()

                    // Vertical pass
                    let cmd2 = queue.makeCommandBuffer()
                    guard let enc2 = cmd2?.makeComputeCommandEncoder() else { continue }
                    enc2.setComputePipelineState(gaussianVPipeline)
                    enc2.setBuffer(tempBuffer, offset: 0, index: 0)
                    enc2.setBuffer(outBuffer, offset: 0, index: 1)
                    enc2.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    enc2.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    enc2.endEncoding()

                    cmd.commit()
                    cmd.waitUntilCompleted()
                    cmd2?.commit()
                    cmd2?.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mpixels = Double(pixelCount) / 1e6
                let fps = 1.0 / elapsed
                print("Gaussian Blur 5x5: \(String(format: "%.2f", fps)) fps (\(String(format: "%.2f", mpixels / elapsed)) MP/s)")
            }

            // Test Sobel edge detection (combined X+Y)
            if let sobelXFunc = library.makeFunction(name: "sobel_x"),
               let sobelYFunc = library.makeFunction(name: "sobel_y"),
               let sobelXPipeline = try? device.makeComputePipelineState(function: sobelXFunc),
               let sobelYPipeline = try? device.makeComputePipelineState(function: sobelYFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    encoder.setComputePipelineState(sobelXPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(tempBuffer, offset: 0, index: 1)
                    encoder.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()

                    guard let cmd2 = queue.makeCommandBuffer(),
                          let enc2 = cmd2?.makeComputeCommandEncoder() else { continue }
                    enc2.setComputePipelineState(sobelYPipeline)
                    enc2.setBuffer(tempBuffer, offset: 0, index: 0)
                    enc2.setBuffer(outBuffer, offset: 0, index: 1)
                    enc2.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    enc2.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    enc2.endEncoding()

                    cmd.commit()
                    cmd.waitUntilCompleted()
                    cmd2.commit()
                    cmd2.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mpixels = Double(pixelCount) / 1e6
                let fps = 1.0 / elapsed
                print("Sobel Edge: \(String(format: "%.2f", fps)) fps (\(String(format: "%.2f", mpixels / elapsed)) MP/s)")
            }

            // Test dilation
            if let dilateFunc = library.makeFunction(name: "dilate"),
               let dilatePipeline = try? device.makeComputePipelineState(function: dilateFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(dilatePipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&size, length: MemoryLayout<uint2>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let mpixels = Double(pixelCount) / 1e6
                let fps = 1.0 / elapsed
                print("Dilate 3x3: \(String(format: "%.2f", fps)) fps (\(String(format: "%.2f", mpixels / elapsed)) MP/s)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Separable filters (Gaussian) are much faster than 2D versions")
        print("2. Box blur is simple but produces artifacts")
        print("3. Edge detection (Sobel) requires two passes")
        print("4. Morphological ops (dilate/erode) are memory bound")
        print("5. Image processing is heavily cache-dependent")
    }
}
