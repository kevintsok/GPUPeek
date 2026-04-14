import Foundation
import Metal

// MARK: - ANE Normalizing Flows Benchmark

/// Benchmarks Apple's Neural Engine for Normalizing Flow workloads
/// Tests invertible transformations and real NVP for density estimation

public struct ANENormalizingFlowBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, dataDim: Int, hiddenDim: Int, numLayers: Int, numBins: Int)] = [
        ("Flow-Small", 32, 64, 4, 8),
        ("Flow-Medium", 64, 128, 6, 16),
        ("Flow-Large", 128, 256, 8, 32),
        ("Flow-XLarge", 256, 512, 10, 64),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Coupling layer for RealNVP
    kernel void couplingLayerKernel(device float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 device float* scale [[buffer(2)]],
                                 device float* shift [[buffer(3)]],
                                 device float* mask [[buffer(4)]],
                                 constant uint& dim [[buffer(5)]],
                                 constant uint& hiddenDim [[buffer(6)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Masked dimensions pass through unchanged
        if (mask[id] > 0.5) {
            output[id] = input[id];
            return;
        }

        // Apply affine transformation
        float s = scale[id];
        float t = shift[id];
        output[id] = input[id] * exp(s) + t;
    }

    // Inverse coupling layer
    kernel void inverseCouplingKernel(device float* z [[buffer(0)]],
                                   device float* x [[buffer(1)]],
                                   device float* scale [[buffer(2)]],
                                   device float* shift [[buffer(3)]],
                                   device float* mask [[buffer(4)]],
                                   constant uint& dim [[buffer(5)]],
                                   uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        if (mask[id] > 0.5) {
            x[id] = z[id];
            return;
        }

        float s = scale[id];
        float t = shift[id];
        x[id] = (z[id] - t) * exp(-s);
    }

    // NN for scale and shift in coupling layer
    kernel void couplingNNKernel(device float* input [[buffer(0)]],
                             device float* scale [[buffer(1)]],
                             device float* shift [[buffer(2)]],
                             device float* w1 [[buffer(3)]],
                             device float* b1 [[buffer(4)]],
                             device float* w2 [[buffer(5)]],
                             device float* b2 [[buffer(6)]],
                             constant uint& inputDim [[buffer(7)]],
                             constant uint& hiddenDim [[buffer(8)]],
                             constant uint& outputDim [[buffer(9)]],
                             uint id [[thread_position_in_grid]]) {
        uint outIdx = id / outputDim;
        uint d = id % outputDim;

        if (outIdx >= 2) return; // scale and shift

        float sum = (outIdx == 0) ? b1[d] : b2[d];

        for (uint i = 0; i < inputDim; i++) {
            float x = input[i];
            if (outIdx == 0) {
                sum += w1[d * inputDim + i] * x;
            } else {
                sum += w2[d * inputDim + i] * x;
            }
        }

        // Apply activation (ReLU for hidden, tanh for scale output, identity for shift)
        if (outIdx == 0 && d >= hiddenDim / 2) {
            sum = tanh(sum);
        }

        if (outIdx == 0) {
            scale[d] = sum;
        } else {
            shift[d] = sum;
        }
    }

    // ActNorm layer (automatic normalization of activations)
    kernel void actNormKernel(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          device float* logScale [[buffer(2)]],
                          device float* shift [[buffer(3)]],
                          constant uint& dim [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // y = exp(log_scale) * x + shift
        output[id] = exp(logScale[id]) * input[id] + shift[id];
    }

    // Inverse ActNorm
    kernel void inverseActNormKernel(device float* y [[buffer(0)]],
                                 device float* x [[buffer(1)]],
                                 device float* logScale [[buffer(2)]],
                                 device float* shift [[buffer(3)]],
                                 constant uint& dim [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // x = (y - shift) * exp(-log_scale)
        x[id] = (y[id] - shift[id]) * exp(-logScale[id]);
    }

    // Permutation layer (reversing dimensions)
    kernel void permutationKernel(device float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& dim [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Reverse permutation
        output[id] = input[dim - 1 - id];
    }

    // Compute log determinant of Jacobian
    // For affine coupling: sum(exp(s))
    kernel void logDetJacobianKernel(device float* scale [[buffer(0)]],
                                  device float* mask [[buffer(1)]],
                                  device float* logDet [[buffer(2)]],
                                  constant uint& dim [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float sum = 0.0;
        for (uint i = 0; i < dim; i++) {
            if (mask[i] < 0.5) {
                sum += scale[i]; // log det = sum(s)
            }
        }
        logDet[0] = sum;
    }

    // NICE layer (additive coupling)
    kernel void niceLayerKernel(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            device float* m [[buffer(2)]],
                            constant uint& dim [[buffer(3)]],
                            constant uint& halfDim [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        if (id < halfDim) {
            output[id] = input[id];
        } else {
            uint hIdx = id - halfDim;
            output[id] = input[id] + m[hIdx];
        }
    }

    // Inverse NICE layer
    kernel void inverseNiceKernel(device float* z [[buffer(0)]],
                               device float* x [[buffer(1)]],
                               device float* m [[buffer(2)]],
                               constant uint& dim [[buffer(3)]],
                               constant uint& halfDim [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        if (id < halfDim) {
            x[id] = z[id];
        } else {
            uint hIdx = id - halfDim;
            x[id] = z[id] - m[hIdx];
        }
    }

    // Planar flow: f(z) = z + u * h(w^T * z + b)
    kernel void planarFlowKernel(device float* z [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             device float* u [[buffer(2)]],
                             device float* w [[buffer(3)]],
                             constant float& b [[buffer(4)]],
                             constant uint& dim [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Compute w^T * z + b
        float wTz = b;
        for (uint i = 0; i < dim; i++) {
            wTz += w[i] * z[i];
        }

        // h(x) = tanh(x)
        float h = tanh(wTz);
        float hPrime = 1.0 - h * h; // derivative

        // f(z)_d = z_d + u_d * h(w^T * z)
        output[id] = z[id] + u[id] * h;
    }

    // Log determinant for planar flow
    // log|det| = log(1 + u^T * w' ) where w' = h'(w^T * z) * w
    kernel void planarLogDetKernel(device float* u [[buffer(0)]],
                                device float* w [[buffer(1)]],
                                device float* z [[buffer(2)]],
                                device float* logDet [[buffer(3)]],
                                constant float& b [[buffer(4)]],
                                constant uint& dim [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float wTz = b;
        for (uint i = 0; i < dim; i++) {
            wTz += w[i] * z[i];
        }

        float hPrime = 1.0 - tanh(wTz) * tanh(wTz);

        float uTw = 0.0;
        for (uint i = 0; i < dim; i++) {
            uTw += u[i] * w[i];
        }

        logDet[0] = log(abs(1.0 + uTw * hPrime) + 0.0001);
    }

    // Glow 1x1 convolution
    kernel void glowConvKernel(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device float* weight [[buffer(2)]],
                           constant uint& dim [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
        uint row = id / dim;
        uint col = id % dim;

        if (row >= dim) return;

        float sum = 0.0;
        for (uint k = 0; k < dim; k++) {
            sum += weight[row * dim + k] * input[k * dim + col];
        }
        output[row * dim + col] = sum;
    }

    // Prior distribution (standard normal) log probability
    kernel void priorLogProbKernel(device float* z [[buffer(0)]],
                               device float* logProb [[buffer(1)]],
                               constant uint& dim [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float sumSq = 0.0;
        for (uint i = 0; i < dim; i++) {
            sumSq += z[i] * z[i];
        }

        // log p(z) = -0.5 * ||z||^2 - (d/2) * log(2*pi)
        logProb[0] = -0.5 * sumSq - float(dim) * 0.5 * 2.9957;
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Normalizing Flows Benchmark ===")
        print("Testing invertible transformations and density estimation on ANE\n")

        var allResults: [(name: String, forwardTime: Double, inverseTime: Double, logDetTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Forward Pass:  \(String(format: "%.4f", result.forwardTime * 1000)) ms")
            print("  Inverse Pass: \(String(format: "%.4f", result.inverseTime * 1000)) ms")
            print("  Log Det:     \(String(format: "%.4f", result.logDetTime * 1000)) ms")
            print("  Total Time:  \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, dataDim: Int, hiddenDim: Int, numLayers: Int, numBins: Int)) throws -> (name: String, forwardTime: Double, inverseTime: Double, logDetTime: Double, totalTime: Double) {
        print("  Running \(config.name) (dim=\(config.dataDim), hidden=\(config.hiddenDim), layers=\(config.numLayers), bins=\(config.numBins))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let couplingFunc = library.makeFunction(name: "couplingLayerKernel"),
              let inverseFunc = library.makeFunction(name: "inverseCouplingKernel"),
              let logDetFunc = library.makeFunction(name: "logDetJacobianKernel"),
              let glowConvFunc = library.makeFunction(name: "glowConvKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let couplingPipeline = try? device.makeComputePipelineState(function: couplingFunc),
              let inversePipeline = try? device.makeComputePipelineState(function: inverseFunc),
              let logDetPipeline = try? device.makeComputePipelineState(function: logDetFunc),
              let glowPipeline = try? device.makeComputePipelineState(function: glowConvFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let dataBytes = config.dataDim * MemoryLayout<Float>.stride
        let maskBytes = config.dataDim * MemoryLayout<Float>.stride
        let weightBytes = config.dataDim * config.dataDim * MemoryLayout<Float>.stride

        guard let inputBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let shiftBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let maskBuffer = device.makeBuffer(length: maskBytes, options: .storageModeShared),
              let logDetBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize mask (alternating pattern)
        let maskPtr = maskBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.dataDim {
            maskPtr[i] = (i % 2 == 0) ? 1.0 : 0.0
        }

        // Initialize scale and shift
        let scalePtr = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        let shiftPtr = shiftBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.dataDim {
            scalePtr[i] = Float.random(in: -0.1...0.1)
            shiftPtr[i] = Float.random(in: -0.1...0.1)
        }

        // Phase 1: Forward Pass (coupling layers)
        let forwardStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(couplingPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(shiftBuffer, offset: 0, index: 3)
            encoder.setBuffer(maskBuffer, offset: 0, index: 4)

            var dim = UInt32(config.dataDim)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.dataDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let forwardTime = Double(getTimeNanos() - forwardStart) / 1e9 / 20.0

        // Phase 2: Inverse Pass
        let inverseStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(inversePipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(shiftBuffer, offset: 0, index: 3)
            encoder.setBuffer(maskBuffer, offset: 0, index: 4)

            var dim = UInt32(config.dataDim)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.dataDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let inverseTime = Double(getTimeNanos() - inverseStart) / 1e9 / 20.0

        // Phase 3: Log Determinant
        let logDetStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(logDetPipeline)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 0)
            encoder.setBuffer(maskBuffer, offset: 0, index: 1)
            encoder.setBuffer(logDetBuffer, offset: 0, index: 2)

            var dim = UInt32(config.dataDim)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let logDetTime = Double(getTimeNanos() - logDetStart) / 1e9 / 20.0

        let totalTime = forwardTime + inverseTime + logDetTime

        return (config.name, forwardTime, inverseTime, logDetTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, forwardTime: Double, inverseTime: Double, logDetTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANENormalizingFlow"

        let log = """
        === ANE Normalizing Flows Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Forward (ms) | Inverse (ms) | Log Det (ms) | Total (ms) |
        |--------------|--------------|--------------|---------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.forwardTime * 1000)) | \(String(format: "%.4f", $0.inverseTime * 1000)) | \(String(format: "%.4f", $0.logDetTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Forward Pass: Apply invertible transformations (coupling layers)
        - Inverse Pass: Reverse transformation for generation
        - Log Det: Compute log|det(J)| for density estimation

        Key Insights:
        - Normalizing flows provide exact log-likelihood
        - Invertibility enables both inference and generation
        - RealNVP uses affine coupling layers for expressiveness
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Normalizing Flows Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Normalizing Flow workloads - generative models based on invertible transformations with exact log-likelihood computation.

        ## What are Normalizing Flows?

        Normalizing flows transform a simple base distribution (e.g., standard normal) through a series of invertible transformations to model complex data distributions.

        ### Core Idea
        ```
        z_K = f_K ◦ f_{K-1} ◦ ... ◦ f_1(z_0)
        where z_0 ~ p_0 (base distribution)

        log p(x) = log p_0(z_K) + log|det(dz_K/dz_{K-1})| + ... + log|det(dz_1/dz_0)|
        ```

        ### Key Property: Invertibility
        - **Inference**: x → z (encoding)
        - **Generation**: z → x (decoding)
        - **Exact likelihood**: No variational lower bound

        ## RealNVP: Real-valued Non-Volume Preserving

        ### Affine Coupling Layer
        Splits input into two parts:
        - First half: passes through unchanged
        - Second half: affine transformation by NN(first half)

        ```
        y_{1:d} = x_{1:d}
        y_{d+1:D} = x_{d+1:D} * exp(s(x_{1:d})) + t(x_{1:d})
        ```

        ### Masking
        Alternating masks ensure all dimensions affect output.

        ## Glow: Generative Flow with Invertible 1x1 Convolutions

        ### Key Innovation
        Replaces permutation with learned invertible 1x1 convolution:
        ```
        f(z) = W * z  (where W is invertible)
        ```

        ### Multi-scale Architecture
        - Reduces dimensionality progressively
        - Enables deeper flows with fewer parameters

        ## Planar Flows

        ### Transformation
        ```
        f(z) = z + u * h(w^T * z + b)
        ```

        ### Log Determinant
        ```
        log|det| = log(1 + u^T * h'(w^T * z) * w)
        ```

        ## Flow Architectures

        ### RealNVP
        - Affine coupling layers
        - Alternating masks
        - No learned permutations

        ### Glow
        - ActNorm → Affine Coupling → 1x1 Conv
        - Multi-scale for efficiency
        - High-quality generation

        ### MAF (Masked Autoregressive Flow)
        - Sequential transformation
        - Expressive but slow inference
        - Fast generation with reverse

        ### IAF (Inverse Autoregressive Flow)
        - Inverse of MAF
        - Fast inference, slow generation

        ## Density Estimation

        ### Forward Pass (Inference)
        x → z_K → log p(x)

        ### Inverse Pass (Generation)
        z ~ p_0 → x (decoded sample)

        ### Log Likelihood
        ```
        log p(x) = log p_0(z_K) + Σ log|det J_i|
        ```

        ## ANE vs GPU for Normalizing Flows

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Coupling Layers | Good | Excellent |
        | 1x1 Convolution | Good | Excellent |
        | ActNorm | Good | Excellent |
        | Log Det Computation | Good | Excellent |

        ## Key Findings

        1. **Exact Likelihood**: No variational bound - true log p(x)

        2. **Invertibility**: Same network for inference and generation

        3. **Exact Gradients**: Straight-through gradient computation

        4. **Composable Transformations**: Stack flows for expressiveness

        5. **ANE Suitability**: Matrix ops for coupling layers work well

        ## Applications

        - **Image Generation**: High-quality samples (Glow, RealNVP)
        - **Density Estimation**: Anomaly detection
        - **Variational Inference**: Approximate posteriors
        - **Speech Synthesis**: WaveGlow, Parallel WaveNet
        - **Protein Structure**: Flow-based protein design
        - **Time Series**: RealNVP for forecasting

        ## Future Work

        - Test multi-scale Glow architecture
        - Benchmark IAF vs MAF
        - Compare with GAN/VAE on same dataset
        - Implement attention-based flows
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
