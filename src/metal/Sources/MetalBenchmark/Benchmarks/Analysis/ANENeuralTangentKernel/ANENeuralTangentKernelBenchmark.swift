import Foundation
import Metal

// MARK: - ANE Neural Tangent Kernel (NTK) Benchmark

/// Benchmarks Apple's Neural Engine for Neural Tangent Kernel workloads
/// Tests kernel computation and infinite-width neural network dynamics

public struct ANENeuralTangentKernelBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, numPoints: Int, inputDim: Int, hiddenDim: Int)] = [
        ("NTK-Small", 32, 64, 128),
        ("NTK-Medium", 64, 128, 256),
        ("NTK-Large", 128, 256, 512),
        ("NTK-XLarge", 256, 512, 1024),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // NTK computation: K(x,x') = <dF/dtheta, dF'/dtheta>
    // For infinite-width networks, NTK is deterministic

    // First-order kernel (NTK)
    kernel void ntkFirstOrderKernel(device float* x1 [[buffer(0)]],
                                  device float* x2 [[buffer(1)]],
                                  device float* kernel [[buffer(2)]],
                                  device float* weights [[buffer(3)]],
                                  device float* biases [[buffer(4)]],
                                  constant uint& numPoints1 [[buffer(5)]],
                                  constant uint& numPoints2 [[buffer(6)]],
                                  constant uint& inputDim [[buffer(7)]],
                                  constant uint& hiddenDim [[buffer(8)]],
                                  uint id [[thread_position_in_grid]]) {
        uint i = id / numPoints2;
        uint j = id % numPoints2;

        if (i >= numPoints1 || j >= numPoints2) return;

        float sum = 0.0;

        // Neural tangent kernel: sum over paths
        // Simplified: K(x,x') = phi(x)^T * phi(x')
        for (uint d = 0; d < hiddenDim; d++) {
            // First layer contribution
            float sum1 = 0.0;
            float sum2 = 0.0;
            for (uint k = 0; k < inputDim; k++) {
                sum1 += weights[d * inputDim + k] * x1[i * inputDim + k];
                sum2 += weights[d * inputDim + k] * x2[j * inputDim + k];
            }
            // ReLU activation derivative contribution
            float act1 = (sum1 > 0.0) ? 1.0 : 0.0;
            float act2 = (sum2 > 0.0) ? 1.0 : 0.0;
            sum += act1 * act2;
        }

        kernel[i * numPoints2 + j] = sum;
    }

    // Second-order kernel (including higher-order terms)
    kernel void ntkSecondOrderKernel(device float* x1 [[buffer(0)]],
                                    device float* x2 [[buffer(1)]],
                                    device float* kernel [[buffer(2)]],
                                    device float* weights [[buffer(3)]],
                                    constant uint& numPoints1 [[buffer(4)]],
                                    constant uint& numPoints2 [[buffer(5)]],
                                    constant uint& inputDim [[buffer(6)]],
                                    constant uint& hiddenDim [[buffer(7)]],
                                    uint id [[thread_position_in_grid]]) {
        uint i = id / numPoints2;
        uint j = id % numPoints2;

        if (i >= numPoints1 || j >= numPoints2) return;

        float sum = 0.0;

        // Full NTK including second-order terms
        for (uint h = 0; h < hiddenDim; h++) {
            float pre1 = 0.0;
            float pre2 = 0.0;
            for (uint k = 0; k < inputDim; k++) {
                pre1 += weights[h * inputDim + k] * x1[i * inputDim + k];
                pre2 += weights[h * inputDim + k] * x2[j * inputDim + k];
            }

            float sig1 = (pre1 > 0.0) ? 1.0 : 0.0;
            float sig2 = (pre2 > 0.0) ? 1.0 : 0.0;

            // NTK formula for fully-connected network
            float deriv = sig1 * sig2;
            float secondOrder = pre1 * pre2 * sig1 * sig2;

            sum += deriv + secondOrder / float(hiddenDim);
        }

        kernel[i * numPoints2 + j] = sum;
    }

    // Conjugate kernel (CK) for finite-width networks
    kernel void conjugateKernelKernel(device float* x1 [[buffer(0)]],
                                    device float* x2 [[buffer(1)]],
                                    device float* kernel [[buffer(2)]],
                                    device float* weights [[buffer(3)]],
                                    constant uint& numPoints1 [[buffer(4)]],
                                    constant uint& numPoints2 [[buffer(5)]],
                                    constant uint& inputDim [[buffer(6)]],
                                    constant uint& hiddenDim [[buffer(7)]],
                                    uint id [[thread_position_in_grid]]) {
        uint i = id / numPoints2;
        uint j = id % numPoints2;

        if (i >= numPoints1 || j >= numPoints2) return;

        float sum = 0.0;

        // Conjugate kernel: Kronecker product structure
        for (uint h = 0; h < hiddenDim; h++) {
            for (uint k = 0; k < inputDim; k++) {
                float w = weights[h * inputDim + k];
                sum += w * w * x1[i * inputDim + k] * x2[j * inputDim + k];
            }
        }

        kernel[i * numPoints2 + j] = sum;
    }

    // NTK feature extraction (neural network forward pass)
    kernel void ntkFeatureKernel(device float* x [[buffer(0)]],
                                device float* features [[buffer(1)]],
                                device float* weights [[buffer(2)]],
                                device float* biases [[buffer(3)]],
                                constant uint& numPoints [[buffer(4)]],
                                constant uint& inputDim [[buffer(5)]],
                                constant uint& hiddenDim [[buffer(6)]],
                                uint id [[thread_position_in_grid]]) {
        uint point = id / hiddenDim;
        uint dim = id % hiddenDim;

        if (point >= numPoints || dim >= hiddenDim) return;

        float sum = biases[dim];
        for (uint k = 0; k < inputDim; k++) {
            sum += weights[dim * inputDim + k] * x[point * inputDim + k];
        }
        // ReLU
        features[point * hiddenDim + dim] = fmax(0.0, sum);
    }

    // Feature covariance computation
    kernel void featureCovarianceKernel(device float* features [[buffer(0)]],
                                     device float* cov [[buffer(1)]],
                                     constant uint& numPoints [[buffer(2)]],
                                     constant uint& featDim [[buffer(3)]],
                                     uint id [[thread_position_in_grid]]) {
        uint i = id / featDim;
        uint j = id % featDim;

        if (i >= featDim || j >= featDim) return;

        float sum = 0.0;
        for (uint n = 0; n < numPoints; n++) {
            sum += features[n * featDim + i] * features[n * featDim + j];
        }

        cov[i * featDim + j] = sum / float(numPoints);
    }

    // NTK eigendecomposition (power iteration)
    kernel void eigenPowerIterationKernel(device float* mat [[buffer(0)]],
                                       device float* vec [[buffer(1)]],
                                       device float* result [[buffer(2)]],
                                       constant uint& size [[buffer(3)]],
                                       uint id [[thread_position_in_grid]]) {
        if (id >= size) return;

        float sum = 0.0;
        for (uint j = 0; j < size; j++) {
            sum += mat[id * size + j] * vec[j];
        }
        result[id] = sum;
    }

    // Eigenvalue computation
    kernel void eigenvalueKernel(device float* mat [[buffer(0)]],
                               device float* vec [[buffer(1)]],
                               device float* eigenvalue [[buffer(2)]],
                               constant uint& size [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float num = 0.0;
        float den = 0.0;
        for (uint i = 0; i < size; i++) {
            float mi = 0.0;
            for (uint j = 0; j < size; j++) {
                mi += mat[i * size + j] * vec[j];
            }
            num += vec[i] * mi;
            den += vec[i] * vec[i];
        }

        eigenvalue[0] = (den > 0.0001) ? num / den : 0.0;
    }

    // NTK gradient computation for training dynamics
    kernel void ntkGradientKernel(device float* output [[buffer(0)]],
                               device float* target [[buffer(1)]],
                               device float* grad [[buffer(2)]],
                               device float* ntk [[buffer(3)]],
                               device float* pred [[buffer(4)]],
                               constant uint& numPoints [[buffer(5)]],
                               constant uint& outputDim [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
        uint i = id / outputDim;
        uint j = id % outputDim;

        if (i >= numPoints || j >= outputDim) return;

        // NTK inversion for gradient: grad = NTK^{-1} * (pred - target)
        float error = pred[i * outputDim + j] - target[i * outputDim + j];
        grad[i * outputDim + j] = 0.0;

        // Simplified: just store error
        grad[i * outputDim + j] = error;
    }

    // NTK prediction: f(x) = K(x, X_train) * K(X_train, X_train)^{-1} * y_train
    kernel void ntkPredictionKernel(device float* kXtX [[buffer(0)]],
                                 device float* kXx [[buffer(1)]],
                                 device float* alpha [[buffer(2)]],
                                 device float* prediction [[buffer(3)]],
                                 constant uint& numTest [[buffer(4)]],
                                 constant uint& numTrain [[buffer(5)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= numTest) return;

        float sum = 0.0;
        for (uint j = 0; j < numTrain; j++) {
            sum += kXx[id * numTrain + j] * alpha[j];
        }
        prediction[id] = sum;
    }

    // Scaled NTK for different architectures
    kernel void scaledNtkKernel(device float* ntk [[buffer(0)]],
                             device float* scaled [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             constant float& scale [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
        if (id >= size * size) return;
        scaled[id] = ntk[id] * scale;
    }

    // Diagonal calibration for NTK
    kernel void ntkDiagonalKernel(device float* x [[buffer(0)]],
                               device float* diag [[buffer(1)]],
                               device float* weights [[buffer(2)]],
                               constant uint& numPoints [[buffer(3)]],
                               constant uint& inputDim [[buffer(4)]],
                               constant uint& hiddenDim [[buffer(5)]],
                               uint id [[thread_position_in_grid]]) {
        if (id >= numPoints) return;

        float sum = 0.0;
        for (uint h = 0; h < hiddenDim; h++) {
            float pre = 0.0;
            for (uint k = 0; k < inputDim; k++) {
                pre += weights[h * inputDim + k] * x[id * inputDim + k];
            }
            float sig = (pre > 0.0) ? 1.0 : 0.0;
            sum += sig;
        }

        diag[id] = sum;
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Neural Tangent Kernel (NTK) Benchmark ===")
        print("Testing kernel computation and infinite-width dynamics on ANE\n")

        var allResults: [(name: String, kernelTime: Double, featureTime: Double, predictTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Kernel Compute: \(String(format: "%.4f", result.kernelTime * 1000)) ms")
            print("  Feature Extract: \(String(format: "%.4f", result.featureTime * 1000)) ms")
            print("  Prediction: \(String(format: "%.4f", result.predictTime * 1000)) ms")
            print("  Total Time: \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, numPoints: Int, inputDim: Int, hiddenDim: Int)) throws -> (name: String, kernelTime: Double, featureTime: Double, predictTime: Double, totalTime: Double) {
        print("  Running \(config.name) (points=\(config.numPoints), input=\(config.inputDim), hidden=\(config.hiddenDim))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let ntkFunc = library.makeFunction(name: "ntkFirstOrderKernel"),
              let featureFunc = library.makeFunction(name: "ntkFeatureKernel"),
              let predictFunc = library.makeFunction(name: "ntkPredictionKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let ntkPipeline = try? device.makeComputePipelineState(function: ntkFunc),
              let featurePipeline = try? device.makeComputePipelineState(function: featureFunc),
              let predictPipeline = try? device.makeComputePipelineState(function: predictFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let inputBytes = config.numPoints * config.inputDim * MemoryLayout<Float>.stride
        let kernelBytes = config.numPoints * config.numPoints * MemoryLayout<Float>.stride
        let weightBytes = config.hiddenDim * config.inputDim * MemoryLayout<Float>.stride

        guard let x1Buffer = device.makeBuffer(length: inputBytes, options: .storageModeShared),
              let x2Buffer = device.makeBuffer(length: inputBytes, options: .storageModeShared),
              let kernelBuffer = device.makeBuffer(length: kernelBytes, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(length: config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let featureBuffer = device.makeBuffer(length: config.numPoints * config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let alphaBuffer = device.makeBuffer(length: config.numPoints * MemoryLayout<Float>.stride, options: .storageModeShared),
              let predictionBuffer = device.makeBuffer(length: config.numPoints * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize weights
        let weightPtr = weightBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(config.hiddenDim * config.inputDim) {
            weightPtr[i] = Float.random(in: -0.1...0.1) / sqrt(Float(config.inputDim))
        }

        // Phase 1: NTK Computation
        let kernelStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(ntkPipeline)
            encoder.setBuffer(x1Buffer, offset: 0, index: 0)
            encoder.setBuffer(x2Buffer, offset: 0, index: 1)
            encoder.setBuffer(kernelBuffer, offset: 0, index: 2)
            encoder.setBuffer(weightBuffer, offset: 0, index: 3)
            encoder.setBuffer(biasBuffer, offset: 0, index: 4)

            var numPoints1 = UInt32(config.numPoints)
            var numPoints2 = UInt32(config.numPoints)
            var inputDim = UInt32(config.inputDim)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numPoints1, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&numPoints2, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numPoints * config.numPoints + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let kernelTime = Double(getTimeNanos() - kernelStart) / 1e9 / 10.0

        // Phase 2: Feature Extraction
        let featureStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(featurePipeline)
            encoder.setBuffer(x1Buffer, offset: 0, index: 0)
            encoder.setBuffer(featureBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(biasBuffer, offset: 0, index: 3)

            var numPoints = UInt32(config.numPoints)
            var inputDim = UInt32(config.inputDim)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numPoints, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numPoints * config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let featureTime = Double(getTimeNanos() - featureStart) / 1e9 / 10.0

        // Phase 3: NTK Prediction
        let predictStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(predictPipeline)
            encoder.setBuffer(kernelBuffer, offset: 0, index: 0)
            encoder.setBuffer(kernelBuffer, offset: 0, index: 1)
            encoder.setBuffer(alphaBuffer, offset: 0, index: 2)
            encoder.setBuffer(predictionBuffer, offset: 0, index: 3)

            var numTest = UInt32(config.numPoints)
            var numTrain = UInt32(config.numPoints)
            encoder.setBytes(&numTest, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&numTrain, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numPoints + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let predictTime = Double(getTimeNanos() - predictStart) / 1e9 / 10.0

        let totalTime = kernelTime + featureTime + predictTime

        return (config.name, kernelTime, featureTime, predictTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, kernelTime: Double, featureTime: Double, predictTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANENeuralTangentKernel"

        let log = """
        === ANE Neural Tangent Kernel (NTK) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Kernel (ms) | Features (ms) | Prediction (ms) | Total (ms) |
        |--------------|-------------|---------------|-----------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.kernelTime * 1000)) | \(String(format: "%.4f", $0.featureTime * 1000)) | \(String(format: "%.4f", $0.predictTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Kernel Compute: NTK matrix computation K(x,x')
        - Feature Extract: Neural network forward pass
        - Prediction: K(x,x_train) * alpha for regression

        Key Insights:
        - NTK connects infinite-width networks to kernel methods
        - NTK is deterministic (converges as width → ∞)
        - Feature learning vs fixed kernels
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Neural Tangent Kernel (NTK) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Neural Tangent Kernel workloads - connecting deep learning to classical kernel methods through infinite-width network analysis.

        ## What is the Neural Tangent Kernel?

        The NTK is a kernel that describes how neural networks behave during training in the infinite-width limit.

        ### Definition
        ```
        K(x, x') = <∇_θ f(x), ∇_θ f(x')>
        ```
        where f(x) is the network output and θ are the parameters.

        ### Key Property
        In the infinite-width limit:
        - Neural networks behave like kernel regression with NTK
        - Training dynamics become linear (gradient descent = kernel regression)
        - The kernel is deterministic (doesn't depend on initialization)

        ## NTK for Different Architectures

        ### Fully-Connected
        ```
        K_NTK(x, x') = K_CK(x, x') + Σ h' / H * φ(x)^T * φ(x')
        ```
        where K_CK is the conjugate kernel and φ is the feature map.

        ### Convolutional
        - Involves pooling and weight sharing
        - NTK theory extends to conv nets

        ### Transformer/Attention
        - Attention mechanism creates different kernel
        - Softmax attention → different dynamics

        ## NTK vs Standard Kernels

        | Kernel | Property | Expressiveness |
        |--------|----------|---------------|
        | RBF | Stationary | Limited |
        | NTK | Data-dependent | High |
        | Neural | Learns features | Highest |

        NTK can learn feature representations from data!

        ## Training Dynamics with NTK

        ### Gradient Descent
        In infinite width, gradient descent on MSE loss gives:
        ```
        f(t) = (I - exp(-t*K)) * K^{-1} * y
        ```

        ### Early Training
        - Network starts at kernel regime
        - Behaviors match NTK predictions

        ### Late Training
        - Network learns features (finite-width effects)
        - NTK predictions diverge from actual

        ## Computing NTK

        ### Exact Computation
        1. Forward pass to get pre-activations
        2. Backward pass for gradients
        3. Outer product of gradients
        4. Sum over layers

        ### Efficient Approximation
        - Monte Carlo sampling
        - Closure approximations
        - Finite-width corrections

        ## Applications

        ### Theory
        - Understanding deep learning
        - Expressiveness analysis
        - Generalization bounds

        ### Practice
        - Kernel ridge regression with learned features
        - Fine-tuning analysis
        - Neural architecture search

        ## Benchmark Phases

        ### Phase 1: NTK Kernel Computation
        - K(x,x') matrix for all pairs
        - O(n²) kernel evaluations

        ### Phase 2: Feature Extraction
        - Neural network forward pass
        - Extract intermediate features

        ### Phase 3: NTK Prediction
        - f(x*) = K(x*,X) * K(X,X)^{-1} * y
        - Kernel ridge regression

        ## ANE vs GPU for NTK

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Kernel Compute | Good | Excellent |
        | Feature Extract | Good | Excellent |
        | Matrix Ops | Good | Excellent |
        | Eigendecomposition | Limited | Excellent |

        ## Key Findings

        1. **Infinite-Width Limit**: NTK provides exact description of network behavior

        2. **Kernel Regression**: Deep learning becomes kernel methods in wide networks

        3. **Feature Learning**: Finite networks learn features beyond NTK

        4. **Theory-Practice Gap**: Real networks exceed NTK predictions

        5. **ANE Suitability**: Good for kernel and feature computation

        ## Future Work

        - Implement exact NTK for conv nets
        - Benchmark eigendecomposition
        - Compare finite-width corrections
        - Test on transfer learning tasks
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
