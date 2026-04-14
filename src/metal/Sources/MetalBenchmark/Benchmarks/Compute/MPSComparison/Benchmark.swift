import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Custom GEMM Kernel (for comparison with MPS)
let gemmShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void gemm_naive(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      constant uint& M [[buffer(3)]],
                      constant uint& N [[buffer(4)]],
                      constant uint& K [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}
"""

public struct MPSComparisonBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Metal Performance Shaders (MPS) vs Custom Kernel Comparison")
        print(String(repeating: "=", count: 70))

        // Create shader library for custom kernels
        guard let library = try? device.makeLibrary(source: gemmShaders, options: nil) else {
            print("Failed to create shader library")
            return
        }

        print("\n--- GEMM (Matrix Multiply) Comparison ---")
        try gemmComparison(library: library)

        print("\n--- Image Convolution Comparison ---")
        try convolutionComparison()
    }

    private func gemmComparison(library: MTLLibrary) throws {
        let M: UInt32 = 1024
        let N: UInt32 = 1024
        let K: UInt32 = 1024
        let iterations = 10

        let aSize = Int(M * K)
        let bSize = Int(K * N)
        let cSize = Int(M * N)

        guard let A = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let B = device.makeBuffer(length: bSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let C = device.makeBuffer(length: cSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize with data
        let aPtr = A.contents().bindMemory(to: Float.self, capacity: aSize)
        let bPtr = B.contents().bindMemory(to: Float.self, capacity: bSize)
        for i in 0..<aSize { aPtr[i] = Float(i % 256) / 256.0 }
        for i in 0..<bSize { bPtr[i] = Float(i % 256) / 256.0 }

        var gflopsNaive: Double = 0

        // 1. Custom Naive GEMM
        if let naiveFunc = library.makeFunction(name: "gemm_naive"),
           let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) {

            var mVal = M, nVal = N, kVal = K
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(naivePipeline)
                encoder.setBuffer(A, offset: 0, index: 0)
                encoder.setBuffer(B, offset: 0, index: 1)
                encoder.setBuffer(C, offset: 0, index: 2)
                encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let flops = 2.0 * Double(M) * Double(N) * Double(K)
            gflopsNaive = flops / elapsed / 1e9
            print("Custom Naive GEMM: \(String(format: "%.2f", gflopsNaive)) GFLOPS")
        }

        print("\n--- Key Insights ---")
        print("1. Custom GEMM shows baseline GPU matrix multiply performance")
        print("2. For production, use MPS or vendor libraries (cuBLAS)")
        print("3. Apple MPS provides optimized GEMM on Apple Silicon")
    }

    private func convolutionComparison() throws {
        let width = 1024
        let height = 1024
        let iterations = 10

        guard let inTexture = device.makeTexture(descriptor: MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false)),
              let outTexture = device.makeTexture(descriptor: MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r32Float, width: width, height: height, mipmapped: false)) else {
            print("Failed to create textures")
            return
        }

        // Fill input texture with data
        let data = (0..<(width * height)).map { Float($0 % 256) / 256.0 }
        inTexture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: data, bytesPerRow: width * MemoryLayout<Float>.size)

        // 1. Custom Sobel (buffer-based)
        let sobelShaders = """
        #include <metal_stdlib>
        using namespace metal;

        constant float sobel_x[9] = { -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f };
        constant float sobel_y[9] = { -1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f };

        kernel void sobel_custom(texture2d<float, access::read> in [[texture(0)]],
                                device float* out [[buffer(0)]],
                                constant uint2& size [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
            if (gid.x >= size.x || gid.y >= size.y) return;

            float gx = 0.0f, gy = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int sx = int(gid.x) + dx;
                    int sy = int(gid.y) + dy;
                    sx = clamp(sx, 0, int(size.x) - 1);
                    sy = clamp(sy, 0, int(size.y) - 1);
                    float val = in.read(uint2(sx, sy)).x;
                    int idx = (dy + 1) * 3 + (dx + 1);
                    gx += val * sobel_x[idx];
                    gy += val * sobel_y[idx];
                }
            }
            out[gid.y * size.x + gid.x] = sqrt(gx * gx + gy * gy);
        }
        """

        guard let sobelLibrary = try? device.makeLibrary(source: sobelShaders, options: nil),
              let sobelFunc = sobelLibrary.makeFunction(name: "sobel_custom"),
              let sobelPipeline = try? device.makeComputePipelineState(function: sobelFunc) else {
            print("Failed to create Sobel pipeline")
            return
        }

        guard let sobelOutBuffer = device.makeBuffer(length: width * height * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeVal = SIMD2<UInt32>(UInt32(width), UInt32(height))

        // Run custom Sobel using texture directly
        let startCustom = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(sobelPipeline)
            encoder.setTexture(inTexture, index: 0)
            encoder.setBuffer(sobelOutBuffer, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endCustom = getTimeNanos()
        let elapsedCustom = getElapsedSeconds(start: startCustom, end: endCustom) / Double(iterations)
        let pixelsPerSecCustom = Double(width * height) / elapsedCustom / 1e6
        print("Custom Sobel (texture): \(String(format: "%.2f", pixelsPerSecCustom)) Mpixels/s")

        // 2. MPS Sobel Filter
        let startMPS = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer() else { continue }

            let sobel = MPSImageSobel(device: device)
            sobel.encode(commandBuffer: cmd, sourceTexture: inTexture, destinationTexture: outTexture)

            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endMPS = getTimeNanos()
        let elapsedMPS = getElapsedSeconds(start: startMPS, end: endMPS) / Double(iterations)
        let pixelsPerSecMPS = Double(width * height) / elapsedMPS / 1e6
        print("MPS Sobel: \(String(format: "%.2f", pixelsPerSecMPS)) Mpixels/s")

        let speedup = pixelsPerSecMPS / pixelsPerSecCustom
        print("\n--- Speedup Analysis ---")
        print("MPS vs Custom: \(String(format: "%.2fx", speedup))")

        print("\n--- Key Insights ---")
        print("1. MPS Sobel uses hardware-optimized texture sampling")
        print("2. Custom Sobel allows flexible filter kernels")
        print("3. For standard filters, MPS provides significant speedup")
        print("4. Custom kernels needed for specialized image processing")
    }
}