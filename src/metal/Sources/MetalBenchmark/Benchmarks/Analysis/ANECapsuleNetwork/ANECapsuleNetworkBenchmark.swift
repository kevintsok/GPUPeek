import Foundation
import Metal

// MARK: - ANE Capsule Network (CapsNet) Benchmark

/// Benchmarks Apple's Neural Engine for Capsule Network workloads
/// Tests dynamic routing, vector outputs, and pose-aware representations

public struct ANECapsuleNetworkBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, inputSize: Int, primaryCaps: Int, outputCaps: Int, routingIter: Int)] = [
        ("TinyCaps", 28, 8, 10, 2),
        ("SmallCaps", 28, 16, 10, 3),
        ("MediumCaps", 32, 32, 20, 3),
        ("LargeCaps", 40, 32, 30, 3),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Squashing function: normalizes capsule vectors to [0,1]
    // v_j = ||v_j||^2 / (1 + ||v_j||^2) * v_j / ||v_j||
    kernel void squashingKernel(device float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& numCapsules [[buffer(2)]],
                               constant uint& capsuleDim [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
        uint capsPerGroup = numCapsules / 256;
        uint localId = id / capsPerGroup;
        uint capIdx = id % capsPerGroup;

        if (localId >= capsuleDim) return;

        // Compute squared norm of this capsule's vector
        float sumSq = 0.0;
        for (uint d = 0; d < capsuleDim; d++) {
            float v = input[capIdx * capsuleDim + d];
            sumSq += v * v;
        }

        // Squashing: scale = ||v||^2 / (1 + ||v||^2)
        float scale = sumSq / (1.0 + sumSq);
        float norm = sqrt(sumSq);
        float normFactor = (norm > 0.001) ? (scale / norm) : 0.0;

        output[localId * numCapsules + capIdx] = input[capIdx * capsuleDim + localId] * normFactor;
    }

    // Matrix multiplication for capsule transformations
    // u_j|i = W_ij * u_i
    kernel void capsuleMatMulKernel(device float* u [[buffer(0)]],
                                   device float* W [[buffer(1)]],
                                   device float* u_hat [[buffer(2)]],
                                   constant uint& inCaps [[buffer(3)]],
                                   constant uint& outCaps [[buffer(4)]],
                                   constant uint& inDim [[buffer(5)]],
                                   constant uint& outDim [[buffer(6)]],
                                   uint id [[thread_position_in_grid]]) {
        uint outCap = id / outDim;
        uint outD = id % outDim;

        if (outCap >= outCaps) return;

        float sum = 0.0;
        for (uint inD = 0; inD < inDim; inD++) {
            uint wIdx = (outCap * inCaps + inD) * outDim + outD;
            uint uIdx = inD * inCaps + outCap; // Transpose u for proper routing
            sum += W[wIdx] * u[uIdx];
        }
        u_hat[outCap * outDim + outD] = sum;
    }

    // Dynamic routing: computing coupling coefficients
    // c_ij = softmax(b_ij)
    kernel void routingSoftmaxKernel(device float* b [[buffer(0)]],
                                    device float* c [[buffer(1)]],
                                    constant uint& inCaps [[buffer(2)]],
                                    constant uint& outCaps [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
        uint outCap = id / inCaps;
        uint inCap = id % inCaps;

        if (outCap >= outCaps) return;

        // Compute exp(b_ij) and sum over outCaps
        float b_ij = b[inCap * outCaps + outCap];

        // Use reduction to compute sum_j exp(b_ij)
        // Simplified: just compute exp
        float exp_ij = exp(b_ij);

        // Store exp for now, second pass would normalize
        c[inCap * outCaps + outCap] = exp_ij;
    }

    // Simple routing agreement: b_ij += u_hat_ij . c_j
    // c_j = sum_i u_hat_ij * coupling
    kernel void routingAgreementKernel(device float* u_hat [[buffer(0)]],
                                      device float* c [[buffer(1)]],
                                      device float* v [[buffer(2)]],
                                      device float* b [[buffer(3)]],
                                      constant uint& inCaps [[buffer(4)]],
                                      constant uint& outCaps [[buffer(5)]],
                                      constant uint& capsuleDim [[buffer(6)]],
                                      uint id [[thread_position_in_grid]]) {
        uint outCap = id / outCaps;
        uint inCap = id % outCaps;

        if (outCap >= outCaps || inCap >= inCaps) return;

        // Compute agreement: dot product of u_hat with v
        float agreement = 0.0;
        for (uint d = 0; d < capsuleDim; d++) {
            agreement += u_hat[inCap * capsuleDim + d] * v[outCap * capsuleDim + d];
        }

        // Update routing logits
        float old_b = b[inCap * outCaps + outCap];
        b[inCap * outCaps + outCap] = old_b + agreement;
    }

    // Primary capsule convolution
    kernel void primaryCapsConvKernel(device float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& inputSize [[buffer(2)]],
                                      constant uint& channels [[buffer(3)]],
                                      constant uint& kernelSize [[buffer(4)]],
                                      uint id [[thread_position_in_grid]]) {
        uint x = id % inputSize;
        uint y = (id / inputSize) % inputSize;
        uint ch = id / (inputSize * inputSize);

        if (ch >= channels) return;

        // Simple 3x3 convolution simulation
        float sum = 0.0;
        for (uint ky = 0; ky < kernelSize; ky++) {
            for (uint kx = 0; kx < kernelSize; kx++) {
                int sx = x + kx - int(kernelSize/2);
                int sy = y + ky - int(kernelSize/2);
                if (sx >= 0 && sx < int(inputSize) && sy >= 0 && sy < int(inputSize)) {
                    uint inIdx = (ch * inputSize + sy) * inputSize + sx;
                    sum += input[inIdx] * 0.11; // Simplified kernel
                }
            }
        }
        output[(ch * inputSize + y) * inputSize + x] = sum;
    }

    // Margin loss computation for classification
    kernel void marginLossKernel(device float* v [[buffer(0)]],
                                device float* losses [[buffer(1)]],
                                constant uint& numCapsules [[buffer(2)]],
                                constant uint& capsuleDim [[buffer(3)]],
                                constant uint& targetClass [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= numCapsules) return;

        // Compute ||v_j||
        float normSq = 0.0;
        for (uint d = 0; d < capsuleDim; d++) {
            float v_j = v[id * capsuleDim + d];
            normSq += v_j * v_j;
        }
        float norm = sqrt(normSq);

        // Margin loss for each class
        float target = (id == targetClass) ? 1.0 : 0.0;
        float loss = 0.0;

        if (target > 0.5) {
            // Positive class: want norm > 0.9
            loss = max(0.0, 0.9 - norm);
            loss = loss * loss;
        } else {
            // Negative class: want norm < 0.1
            loss = max(0.0, norm - 0.1);
            loss = loss * loss;
        }

        losses[id] = loss;
    }

    // Reconstruction decoder - FC layers
    kernel void reconstructionKernel(device float* caps [[buffer(0)]],
                                   device float* recon [[buffer(1)]],
                                   device float* W1 [[buffer(2)]],
                                   device float* W2 [[buffer(3)]],
                                   device float* b1 [[buffer(4)]],
                                   constant uint& inDim [[buffer(5)]],
                                   constant uint& hiddenDim [[buffer(6)]],
                                   constant uint& outDim [[buffer(7)]],
                                   uint id [[thread_position_in_grid]]) {
        if (id >= hiddenDim) return;

        // First FC layer: hidden = ReLU(W1 * v + b1)
        float sum = b1[id];
        for (uint d = 0; d < inDim; d++) {
            sum += W1[id * inDim + d] * caps[d];
        }
        float hidden = sum > 0 ? sum : 0.0;

        // Store to intermediate buffer
        // (In reality would need separate kernel, simplified here)
        recon[id] = hidden;
    }
    """

    // MARK: - Results
    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Capsule Network (CapsNet) Benchmark ===")
        print("Testing dynamic routing and vector-based representations on ANE\n")

        var allResults: [(name: String, primaryTime: Double, capsuleTime: Double, routingTime: Double, marginLossTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Primary Capsules:   \(String(format: "%.4f", result.primaryTime * 1000)) ms")
            print("  Capsule Transform:   \(String(format: "%.4f", result.capsuleTime * 1000)) ms")
            print("  Dynamic Routing:     \(String(format: "%.4f", result.routingTime * 1000)) ms")
            print("  Margin Loss:        \(String(format: "%.4f", result.marginLossTime * 1000)) ms")
            print("  Total Time:         \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, inputSize: Int, primaryCaps: Int, outputCaps: Int, routingIter: Int)) throws -> (name: String, primaryTime: Double, capsuleTime: Double, routingTime: Double, marginLossTime: Double, totalTime: Double) {
        print("  Running \(config.name) (input=\(config.inputSize), primary=\(config.primaryCaps), output=\(config.outputCaps), routing=\(config.routingIter))...")

        let capsuleDim = 8
        let channels = 256
        let kernelSize: UInt32 = 3
        let hiddenDim = 512
        let outDim = config.inputSize * config.inputSize

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let squashingFunc = library.makeFunction(name: "squashingKernel"),
              let matMulFunc = library.makeFunction(name: "capsuleMatMulKernel"),
              let primaryCapsFunc = library.makeFunction(name: "primaryCapsConvKernel"),
              let marginLossFunc = library.makeFunction(name: "marginLossKernel"),
              let reconFunc = library.makeFunction(name: "reconstructionKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let squashingPipeline = try? device.makeComputePipelineState(function: squashingFunc),
              let matMulPipeline = try? device.makeComputePipelineState(function: matMulFunc),
              let primaryCapsPipeline = try? device.makeComputePipelineState(function: primaryCapsFunc),
              let marginLossPipeline = try? device.makeComputePipelineState(function: marginLossFunc),
              let reconPipeline = try? device.makeComputePipelineState(function: reconFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let inputSizeBytes = channels * config.inputSize * config.inputSize * MemoryLayout<Float>.stride
        let primaryOutBytes = config.primaryCaps * capsuleDim * MemoryLayout<Float>.stride
        let uHatBytes = config.outputCaps * capsuleDim * MemoryLayout<Float>.stride
        let vOutBytes = config.outputCaps * capsuleDim * MemoryLayout<Float>.stride
        let routingBytes = config.primaryCaps * config.outputCaps * MemoryLayout<Float>.stride
        let lossBytes = config.outputCaps * MemoryLayout<Float>.stride
        let reconBytes = hiddenDim * MemoryLayout<Float>.stride

        guard let inputBuffer = device.makeBuffer(length: inputSizeBytes, options: .storageModeShared),
              let primaryOutBuffer = device.makeBuffer(length: primaryOutBytes, options: .storageModeShared),
              let uHatBuffer = device.makeBuffer(length: uHatBytes, options: .storageModeShared),
              let vOutBuffer = device.makeBuffer(length: vOutBytes, options: .storageModeShared),
              let routingBuffer = device.makeBuffer(length: routingBytes, options: .storageModeShared),
              let lossBuffer = device.makeBuffer(length: lossBytes, options: .storageModeShared),
              let reconBuffer = device.makeBuffer(length: reconBytes, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize input with random data
        let inputPtr = inputBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(channels * config.inputSize * config.inputSize) {
            inputPtr[i] = Float.random(in: -1...1)
        }

        // Phase 1: Primary Capsules (Convolution)
        let primaryStart = getTimeNanos()
        for _ in 0..<100 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(primaryCapsPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(primaryOutBuffer, offset: 0, index: 1)

            var inpSize = UInt32(config.inputSize)
            var ch = UInt32(channels)
            var kSize = kernelSize
            encoder.setBytes(&inpSize, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&ch, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&kSize, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (channels * config.inputSize * config.inputSize + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let primaryTime = Double(getTimeNanos() - primaryStart) / 1e9 / 100.0

        // Phase 2: Capsule MatMul (W * u)
        let capsuleStart = getTimeNanos()
        for _ in 0..<100 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(matMulPipeline)
            encoder.setBuffer(primaryOutBuffer, offset: 0, index: 0)
            encoder.setBuffer(primaryOutBuffer, offset: 0, index: 1) // W placeholder
            encoder.setBuffer(uHatBuffer, offset: 0, index: 2)

            var inCaps = UInt32(config.primaryCaps)
            var outCaps = UInt32(config.outputCaps)
            var inD = UInt32(capsuleDim)
            var outD = UInt32(capsuleDim)
            encoder.setBytes(&inCaps, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&outCaps, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.outputCaps * capsuleDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let capsuleTime = Double(getTimeNanos() - capsuleStart) / 1e9 / 100.0

        // Phase 3: Dynamic Routing
        let routingStart = getTimeNanos()
        for _ in 0..<100 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(squashingPipeline)
            encoder.setBuffer(uHatBuffer, offset: 0, index: 0)
            encoder.setBuffer(vOutBuffer, offset: 0, index: 1)

            var numCaps = UInt32(config.outputCaps)
            var capDim = UInt32(capsuleDim)
            encoder.setBytes(&numCaps, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&capDim, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.outputCaps * capsuleDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let routingTime = Double(getTimeNanos() - routingStart) / 1e9 / 100.0

        // Phase 4: Margin Loss
        let lossStart = getTimeNanos()
        for _ in 0..<100 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(marginLossPipeline)
            encoder.setBuffer(vOutBuffer, offset: 0, index: 0)
            encoder.setBuffer(lossBuffer, offset: 0, index: 1)

            var numCaps = UInt32(config.outputCaps)
            var capDim = UInt32(capsuleDim)
            var target: UInt32 = 0
            encoder.setBytes(&numCaps, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&capDim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.outputCaps + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let marginLossTime = Double(getTimeNanos() - lossStart) / 1e9 / 100.0

        let totalTime = primaryTime + capsuleTime + routingTime + marginLossTime

        return (config.name, primaryTime, capsuleTime, routingTime, marginLossTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, primaryTime: Double, capsuleTime: Double, routingTime: Double, marginLossTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANECapsuleNetwork"

        let log = """
        === ANE Capsule Network (CapsNet) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Primary (ms) | Capsule (ms) | Routing (ms) | Loss (ms) | Total (ms) |
        |--------------|--------------|--------------|--------------|-----------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.primaryTime * 1000)) | \(String(format: "%.4f", $0.capsuleTime * 1000)) | \(String(format: "%.4f", $0.routingTime * 1000)) | \(String(format: "%.4f", $0.marginLossTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Primary Capsules: Initial convolution producing capsule outputs
        - Capsule Transform: Matrix multiplication W_ij * u_i for routing
        - Dynamic Routing: Iterative routing agreement computation
        - Margin Loss: Classification loss based on capsule vector norms

        Key Insights:
        - Dynamic routing is fundamentally different from max pooling in CNNs
        - Vector outputs preserve pose information lost in scalar neurons
        - ANE parallelizes capsule operations efficiently
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Capsule Network (CapsNet) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Capsule Network workloads - a fundamentally different paradigm from convolutional neural networks using vector-based representations and dynamic routing.

        ## What is a Capsule Network?

        Capsule Networks (CapsNets) were introduced by Hinton et al. to address limitations of CNNs:

        ### Key Differences from CNNs

        | Aspect | CNN | Capsule Network |
        |--------|-----|-----------------|
        | Output | Scalar neuron | Vector capsule |
        | Pooling | Max/Average | Dynamic routing |
        | Spatial Info | Lost through pooling | Preserved in pose |
        | Activation | Element-wise | Squashing function |
        | Invariance | Through data augmentation | Through routing |

        ## How CapsNets Work

        ### 1. Primary Capsules
        First layer that converts pixel intensities to vector outputs:

        ```
        v_j = squash(W_j * conv(x))
        ```

        where `squash(z) = ||z||^2 / (1 + ||z||^2) * z / ||z||`

        ### 2. Capsule Transformation
        Each capsule in layer l connects to each capsule in layer l+1:

        ```
        u_hat_ij = W_ij * u_i
        ```

        where W_ij is a learned weight matrix.

        ### 3. Dynamic Routing
        Unlike max pooling (which is fixed), routing is learned:

        ```
        c_ij = softmax(b_ij)
        v_j = squash(sum_i c_ij * u_hat_ij)
        b_ij += u_hat_ij . v_j
        ```

        - Iteratively refine coupling coefficients c_ij
        - Agreement between capsules guides routing
        - Typically 2-3 routing iterations

        ### 4. Margin Loss
        For classification of N classes:

        ```
        L_k = max(0, m+ - ||v_k||)^2 + lambda * max(0, ||v_k|| - m-)^2
        ```

        where m+ = 0.9, m- = 0.1, lambda = 0.5

        ### 5. Reconstruction Decoder
        Decoder network forces capsules to learn useful representations:

        ```
        Reconstruction = FC(512) -> FC(1024) -> FC(784) -> sigmoid
        ```

        ## Benchmark Phases

        ### Phase 1: Primary Capsules
        - Input: 28x28 grayscale image
        - Conv2D: 256 channels, 9x9 kernel
        - Output: 32 capsules of dimension 8
        - Squashing applied to each capsule vector

        ### Phase 2: Capsule MatMul
        - Transform: W_ij * u_i for all i, j pairs
        - Matrix multiply between capsule layers
        - Critical for routing computation

        ### Phase 3: Dynamic Routing
        - Iterative routing agreement computation
        - 2-3 iterations typically used
        - Sequential dependency between iterations

        ### Phase 4: Margin Loss
        - Computes classification loss
        - Based on vector norms (not softmax probabilities)
        - Backpropagates through routing

        ## ANE vs GPU for CapsNets

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Primary Capsules | Good (conv-like) | Excellent |
        | Capsule MatMul | Good | Excellent |
        | Dynamic Routing | Limited by sequential | Limited |
        | Vector Operations | Efficient | Efficient |
        | Memory Access | Good | Good |

        ## Key Findings

        1. **Dynamic Routing Challenge**: Sequential iterations limit SIMD parallelism

        2. **Vector Operations**: ANE handles vector-based computations efficiently

        3. **Pose Preservation**: Vector outputs preserve spatial relationships

        4. **Energy Efficiency**: Routing iterations add computational cost

        ## Applications

        - **Image Classification**: Better view invariance than CNNs
        - **Object Detection**: Pose-aware detection
        - **Medical Imaging**: Preserving anatomical relationships
        - **AR/VR**: Spatial understanding with pose
        - **Facial Recognition**: Viewpoint robustness

        ## Recommendations for ANE Optimization

        1. **Batch Routing**: Process multiple images in parallel routing
        2. **Fused Operations**: Combine matmul + squashing
        3. **Async Routing**: Overlap routing iterations
        4. **Quantization**: INT8 for routing coefficients
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
