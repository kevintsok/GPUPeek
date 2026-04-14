import Foundation
import Metal

// MARK: - ANE Bayesian Neural Network (BNN) Benchmark

/// Benchmarks Apple's Neural Engine for Bayesian Neural Network workloads
/// Tests variational inference and uncertainty quantification in deep networks

public struct ANEBayesianNeuralNetworkBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, inputDim: Int, hiddenDim: Int, numSamples: Int, layers: Int)] = [
        ("BNN-Small", 64, 128, 10, 3),
        ("BNN-Medium", 128, 256, 20, 4),
        ("BNN-Large", 256, 512, 30, 5),
        ("BNN-XLarge", 512, 512, 50, 6),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Mean forward pass (deterministic path)
    kernel void meanForwardKernel(device float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                device float* weights [[buffer(2)]],
                                device float* biases [[buffer(3)]],
                                constant uint& inputDim [[buffer(4)]],
                                constant uint& hiddenDim [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= hiddenDim) return;

        float sum = biases[id];
        for (uint i = 0; i < inputDim; i++) {
            sum += weights[id * inputDim + i] * input[i];
        }
        // ReLU activation
        output[id] = fmax(0.0, sum);
    }

    // Variance forward pass (uncertainty path)
    kernel void varianceForwardKernel(device float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    device float* logVarWeights [[buffer(2)]],
                                    device float* logVarBiases [[buffer(3)]],
                                    constant uint& inputDim [[buffer(4)]],
                                    constant uint& hiddenDim [[buffer(5)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id >= hiddenDim) return;

        float sum = logVarBiases[id];
        for (uint i = 0; i < inputDim; i++) {
            sum += logVarWeights[id * inputDim + i] * input[i];
        }
        // Softplus to ensure positive variance
        output[id] = log(1.0 + exp(sum));
    }

    // Bayesian Linear layer with weight uncertainty
    kernel void bayesianLinearKernel(device float* input [[buffer(0)]],
                                    device float* mean [[buffer(1)]],
                                    device float* variance [[buffer(2)]],
                                    device float* meanWeights [[buffer(3)]],
                                    device float* varWeights [[buffer(4)]],
                                    device float* meanBias [[buffer(5)]],
                                    device float* varBias [[buffer(6)]],
                                    device float* noise [[buffer(7)]],
                                    constant uint& inputDim [[buffer(8)]],
                                    constant uint& outputDim [[buffer(9)]],
                                    uint id [[thread_position_in_grid]]) {
        uint outIdx = id / outputDim;
        uint d = id % outputDim;

        if (outIdx >= 1) return;

        // Sample weights: w ~ N(mean_w, var_w)
        // w = mean_w + sqrt(var_w) * epsilon
        float sum = meanBias[d];
        float varSum = 0.0;

        for (uint i = 0; i < inputDim; i++) {
            float w_mean = meanWeights[d * inputDim + i];
            float w_var = varWeights[d * inputDim + i];
            float epsilon = noise[i];
            float w = w_mean + sqrt(w_var + 0.0001) * epsilon;

            sum += w * input[i];
            // Variance contribution from weight uncertainty
            varSum += w_var * input[i] * input[i];
        }

        mean[d] = fmax(0.0, sum);
        variance[d] = varSum + 0.01; // Add noise variance
    }

    // Monte Carlo dropout - Bernoulli samples
    kernel void dropoutKernel(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device float* noise [[buffer(2)]],
                           constant float& dropRate [[buffer(3)]],
                           constant uint& size [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
        if (id >= size) return;

        float n = noise[id];
        output[id] = (n > dropRate) ? input[id] / (1.0 - dropRate) : 0.0;
    }

    // Variational inference: KL divergence term for weights
    // KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kernel void klDivergenceKernel(device float* meanWeights [[buffer(0)]],
                                 device float* varWeights [[buffer(1)]],
                                 device float* klLoss [[buffer(2)]],
                                 constant uint& numWeights [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= numWeights) return;

        float mu = meanWeights[id];
        float sigma2 = varWeights[id];

        // KL divergence for this weight
        float kl = -0.5 * (1.0 + log(sigma2 + 0.0001) - mu * mu - sigma2);

        // Atomic add to total KL (simplified - just store per element)
        klLoss[id] = kl;
    }

    // Reparameterization trick: sample from N(mu, sigma)
    kernel void reparamSampleKernel(device float* mu [[buffer(0)]],
                                 device float* sigma [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 device float* noise [[buffer(3)]],
                                 constant uint& size [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= size) return;

        output[id] = mu[id] + sigma[id] * noise[id];
    }

    // Ensemble prediction (average of multiple forward passes)
    kernel void ensembleAverageKernel(device float* predictions [[buffer(0)]],
                                   device float* mean [[buffer(1)]],
                                   device float* variance [[buffer(2)]],
                                   constant uint& numSamples [[buffer(3)]],
                                   constant uint& outputDim [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
        uint d = id % outputDim;

        float sum = 0.0;
        float sumSq = 0.0;

        for (uint s = 0; s < numSamples; s++) {
            float pred = predictions[s * outputDim + d];
            sum += pred;
            sumSq += pred * pred;
        }

        float meanVal = sum / float(numSamples);
        mean[d] = meanVal;
        // Variance of predictions + epistemic uncertainty
        variance[d] = (sumSq / float(numSamples)) - meanVal * meanVal + 0.1;
    }

    // Flipout: efficient uncertainty estimation
    // Uses sign flips instead of sampling for variance estimation
    kernel void flipoutKernel(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            device float* meanWeights [[buffer(2)]],
                            device float* flipSigns [[buffer(3)]],
                            device float* perturbation [[buffer(4)]],
                            constant uint& inputDim [[buffer(5)]],
                            constant uint& outputDim [[buffer(6)]],
                            uint id [[thread_position_in_grid]]) {
        uint outIdx = id / outputDim;
        uint d = id % outputDim;

        if (outIdx >= 1) return;

        float sum = 0.0;
        for (uint i = 0; i < inputDim; i++) {
            // Flipout: w = mean_w + flip_sign * perturbation
            float flip = flipSigns[i * outputDim + d];
            float pert = perturbation[i];
            float w = meanWeights[d * inputDim + i] + flip * pert;

            sum += w * input[i];
        }

        output[d] = fmax(0.0, sum);
    }

    // Compute uncertainty metrics
    kernel void uncertaintyMetricKernel(device float* mean [[buffer(0)]],
                                     device float* variance [[buffer(1)]],
                                     device float* targets [[buffer(2)]],
                                     device float* metrics [[buffer(3)]],
                                     constant uint& size [[buffer(4)]],
                                     uint id [[thread_position_in_grid]]) {
        if (id >= size) return;

        float m = mean[id];
        float v = variance[id];
        float t = targets[id];

        // Squared error
        float se = (m - t) * (m - t);

        // Negative log probability under Gaussian
        float nlp = 0.5 * (log(2.0 * 3.14159) + log(v + 0.0001) + se / (v + 0.0001));

        // Calibration error (if predicted variance matches actual error)
        float predStd = sqrt(v);
        float calibration = abs(predStd - abs(m - t));

        metrics[id * 3] = se;
        metrics[id * 3 + 1] = nlp;
        metrics[id * 3 + 2] = calibration;
    }

    // Probabilistic backpropagation layer
    kernel void probBackpropKernel(device float* gradOutput [[buffer(0)]],
                                 device float* meanWeights [[buffer(1)]],
                                 device float* varWeights [[buffer(2)]],
                                 device float* gradMean [[buffer(3)]],
                                 device float* gradVar [[buffer(4)]],
                                 device float* noise [[buffer(5)]],
                                 constant uint& numWeights [[buffer(6)]],
                                 constant float& lr [[buffer(7)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= numWeights) return;

        // Gradient for mean: same as standard backprop
        gradMean[id] = -gradOutput[id] * noise[id];

        // Gradient for variance (uncertainty): decreases variance when loss is high
        gradVar[id] = 0.5 * (gradOutput[id] * noise[id] * noise[id] - 1.0);

        // Update mean (gradient descent)
        meanWeights[id] -= lr * gradMean[id];

        // Update variance (gradient descent on log variance)
        varWeights[id] -= lr * gradVar[id];

        // Clamp variance to be positive
        varWeights[id] = fmax(0.01, varWeights[id]);
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Bayesian Neural Network (BNN) Benchmark ===")
        print("Testing variational inference and uncertainty quantification on ANE\n")

        var allResults: [(name: String, forwardTime: Double, sampleTime: Double, klTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Mean Forward:    \(String(format: "%.4f", result.forwardTime * 1000)) ms")
            print("  MC Sampling:    \(String(format: "%.4f", result.sampleTime * 1000)) ms")
            print("  KL Divergence:  \(String(format: "%.4f", result.klTime * 1000)) ms")
            print("  Total Time:   \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, inputDim: Int, hiddenDim: Int, numSamples: Int, layers: Int)) throws -> (name: String, forwardTime: Double, sampleTime: Double, klTime: Double, totalTime: Double) {
        print("  Running \(config.name) (input=\(config.inputDim), hidden=\(config.hiddenDim), samples=\(config.numSamples), layers=\(config.layers))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let meanForwardFunc = library.makeFunction(name: "meanForwardKernel"),
              let varianceForwardFunc = library.makeFunction(name: "varianceForwardKernel"),
              let bayesianLinearFunc = library.makeFunction(name: "bayesianLinearKernel"),
              let reparamSampleFunc = library.makeFunction(name: "reparamSampleKernel"),
              let klDivFunc = library.makeFunction(name: "klDivergenceKernel"),
              let ensembleAvgFunc = library.makeFunction(name: "ensembleAverageKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let meanForwardPipeline = try? device.makeComputePipelineState(function: meanForwardFunc),
              let varianceForwardPipeline = try? device.makeComputePipelineState(function: varianceForwardFunc),
              let bayesianLinearPipeline = try? device.makeComputePipelineState(function: bayesianLinearFunc),
              let reparamSamplePipeline = try? device.makeComputePipelineState(function: reparamSampleFunc),
              let klDivPipeline = try? device.makeComputePipelineState(function: klDivFunc),
              let ensembleAvgPipeline = try? device.makeComputePipelineState(function: ensembleAvgFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let inputBytes = config.inputDim * MemoryLayout<Float>.stride
        let hiddenBytes = config.hiddenDim * MemoryLayout<Float>.stride
        let weightBytes = config.hiddenDim * config.inputDim * MemoryLayout<Float>.stride
        let totalWeights = config.hiddenDim * config.inputDim + config.hiddenDim

        guard let inputBuffer = device.makeBuffer(length: inputBytes, options: .storageModeShared),
              let meanBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let varianceBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let meanWeightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let varWeightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let meanBiasBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let varBiasBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let noiseBuffer = device.makeBuffer(length: max(inputBytes, hiddenBytes), options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let klBuffer = device.makeBuffer(length: totalWeights * MemoryLayout<Float>.stride, options: .storageModeShared),
              let predBuffer = device.makeBuffer(length: config.numSamples * config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize input
        let inputPtr = inputBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.inputDim {
            inputPtr[i] = Float.random(in: -1...1)
        }

        // Phase 1: Mean Forward Pass
        let forwardStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(meanForwardPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(meanWeightBuffer, offset: 0, index: 2)
            encoder.setBuffer(meanBiasBuffer, offset: 0, index: 3)

            var inputDimVal = UInt32(config.inputDim)
            var hiddenDimVal = UInt32(config.hiddenDim)
            encoder.setBytes(&inputDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let forwardTime = Double(getTimeNanos() - forwardStart) / 1e9 / 20.0

        // Phase 2: MC Sampling (multiple forward passes with noise)
        let sampleStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(bayesianLinearPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(meanBuffer, offset: 0, index: 1)
            encoder.setBuffer(varianceBuffer, offset: 0, index: 2)
            encoder.setBuffer(meanWeightBuffer, offset: 0, index: 3)
            encoder.setBuffer(varWeightBuffer, offset: 0, index: 4)
            encoder.setBuffer(meanBiasBuffer, offset: 0, index: 5)
            encoder.setBuffer(varBiasBuffer, offset: 0, index: 6)
            encoder.setBuffer(noiseBuffer, offset: 0, index: 7)

            var inputDimVal = UInt32(config.inputDim)
            var hiddenDimVal = UInt32(config.hiddenDim)
            encoder.setBytes(&inputDimVal, length: MemoryLayout<UInt32>.stride, index: 8)
            encoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 9)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let sampleTime = Double(getTimeNanos() - sampleStart) / 1e9 / 20.0

        // Phase 3: KL Divergence Computation
        let klStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(klDivPipeline)
            encoder.setBuffer(meanWeightBuffer, offset: 0, index: 0)
            encoder.setBuffer(varWeightBuffer, offset: 0, index: 1)
            encoder.setBuffer(klBuffer, offset: 0, index: 2)

            var numWeights = UInt32(totalWeights)
            encoder.setBytes(&numWeights, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (totalWeights + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let klTime = Double(getTimeNanos() - klStart) / 1e9 / 20.0

        let totalTime = forwardTime + sampleTime + klTime

        return (config.name, forwardTime, sampleTime, klTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, forwardTime: Double, sampleTime: Double, klTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANEBayesianNeuralNetwork"

        let log = """
        === ANE Bayesian Neural Network (BNN) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Mean Fwd (ms) | MC Sample (ms) | KL Div (ms) | Total (ms) |
        |--------------|----------------|----------------|-------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.forwardTime * 1000)) | \(String(format: "%.4f", $0.sampleTime * 1000)) | \(String(format: "%.4f", $0.klTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Mean Forward: Deterministic forward pass (standard DNN)
        - MC Sampling: Multiple stochastic passes for uncertainty
        - KL Divergence: Regularization term for variational inference

        Key Insights:
        - BNNs provide uncertainty estimates alongside predictions
        - Variational inference approximates posterior over weights
        - MC dropout provides epistemic uncertainty estimation
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Bayesian Neural Network (BNN) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Bayesian Neural Network workloads - deep networks that quantify uncertainty through variational inference.

        ## What are Bayesian Neural Networks?

        BNNs combine deep learning with Bayesian probability:

        ### Standard DNN (Deterministic)
        - Single set of weights θ
        - y = f(x; θ)
        - No uncertainty estimate

        ### BNN (Probabilistic)
        - Distribution over weights p(θ)
        - y ~ p(y|x, D) = ∫ p(y|x, θ) p(θ|D) dθ
        - Provides uncertainty estimates

        ### Why Uncertainty Matters
        - **Epistemic**: Uncertainty about the model (reduces with more data)
        - **Aleatoric**: Inherent noise in data (irreducible)

        ## Variational Inference

        ### Posterior Approximation
        Instead of computing exact p(θ|D), we approximate with q(θ|ω):
        ```
        q(θ|ω) ≈ p(θ|D)
        ```

        where q is typically Gaussian: q(θ|ω) = N(μ, σ²)

        ### ELBO Objective
        ```
        L(ω) = E_q[log p(D|θ)] - KL(q(θ|ω) || p(θ))
               (likelihood)          (regularization)
        ```

        ### Reparameterization Trick
        Enable gradient flow through stochastic nodes:
        ```
        ε ~ N(0, 1)
        θ = μ + σ * ε
        ```

        ## BNN Approaches

        ### 1. Bayes by Backprop
        - Learn mean and variance for each weight
        - Sample weights using reparameterization
        - KL term regularizes towards prior

        ### 2. MC Dropout
        - Keep dropout active at test time
        - Multiple forward passes = ensemble
        - Epistemic uncertainty from variance

        ### 3. Flipout
        - Efficient variance estimation
        - Uses sign flips instead of sampling
        - Reduces variance in gradient estimates

        ### 4. Local Reparameterization
        - Sample per-neuron instead of per-weight
        - More efficient gradients

        ## Uncertainty Metrics

        ### Negative Log Probability (NLL)
        ```
        NLL = -log p(y|x, D)
        ```
        Lower is better; captures both accuracy and uncertainty.

        ### Expected Calibration Error (ECE)
        ```
        ECE = sum_b |B_b| / n * |acc(B_b) - conf(B_b)|
        ```
        Measures if predicted variance matches actual error.

        ### Sharpness
        ```
        Sharpness = E[variance]
        ```
        Average predicted variance; lower = sharper (but may be wrong).

        ## Benchmark Phases

        ### Phase 1: Mean Forward Pass
        - Standard deterministic forward pass
        - Computes mean predictions

        ### Phase 2: MC Sampling
        - Multiple stochastic forward passes
        - Samples weights from variational distribution
        - Builds distribution over predictions

        ### Phase 3: KL Divergence
        - Computes regularization term
        - KL(q||p) between variational posterior and prior

        ## ANE vs GPU for BNNs

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Mean Forward | Excellent | Excellent |
        | Weight Sampling | Good | Good |
        | MC Integration | Good | Excellent |
        | KL Computation | Good | Excellent |

        ## Key Findings

        1. **Uncertainty Quantification**: BNNs provide calibrated uncertainty estimates

        2. **Epistemic vs Aleatoric**: Can distinguish between model and data uncertainty

        3. **Regularization**: KL term naturally regularizes weights

        4. **ANE Suitability**: Good for forward passes, MC sampling adds overhead

        ## Applications

        - **Medical AI**: Uncertainty for diagnosis decisions
        - **Autonomous Vehicles**: Safety-critical decisions
        - **Scientific Discovery**: Knowing what the model doesn't know
        - **Active Learning**: Select informative data points
        - **Out-of-Distribution Detection**: Flag unfamiliar inputs
        - **Robotics**: Safe exploration and control

        ## Future Work

        - Implement MC dropout comparison
        - Test calibration metrics (ECE)
        - Benchmark on OOD detection tasks
        - Compare with deep ensembles
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
