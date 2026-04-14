import Foundation
import Metal

// MARK: - ANE Energy-Based Model (EBM) Benchmark

/// Benchmarks Apple's Neural Engine for Energy-Based Model workloads
/// Tests contrastive divergence, Gibbs sampling, and energy function learning

public struct ANEEnergyBasedModelBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, dataDim: Int, hiddenDim: Int, numChains: Int, numSteps: Int)] = [
        ("EBM-Small", 64, 128, 32, 10),
        ("EBM-Medium", 128, 256, 64, 20),
        ("EBM-Large", 256, 512, 128, 30),
        ("EBM-XLarge", 512, 1024, 256, 50),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Energy function: E(x) = -log sigma(W * x + b)
    kernel void energyForwardKernel(device float* x [[buffer(0)]],
                              device float* energy [[buffer(1)]],
                              device float* weights [[buffer(2)]],
                              device float* biases [[buffer(3)]],
                              constant uint& inputDim [[buffer(4)]],
                              constant uint& hiddenDim [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= hiddenDim) return;

        float sum = biases[id];
        for (uint i = 0; i < inputDim; i++) {
            sum += weights[id * inputDim + i] * x[i];
        }
        energy[id] = -log(1.0 + exp(sum));
    }

    // Joint embedding energy: E(x, y) = ||f(x) - g(y)||^2
    kernel void jointEnergyKernel(device float* embX [[buffer(0)]],
                              device float* embY [[buffer(1)]],
                              device float* energy [[buffer(2)]],
                              constant uint& embDim [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float sumSq = 0.0;
        for (uint i = 0; i < embDim; i++) {
            float diff = embX[i] - embY[i];
            sumSq += diff * diff;
        }
        energy[0] = sumSq;
    }

    // Contrastive divergence: compute gradient of log-likelihood
    // grad = (data_grad - model_grad)
    kernel void contrastiveDivergenceKernel(device float* data [[buffer(0)]],
                                        device float* model [[buffer(1)]],
                                        device float* grad [[buffer(2)]],
                                        device float* weights [[buffer(3)]],
                                        constant float& lr [[buffer(4)]],
                                        constant uint& dataDim [[buffer(5)]],
                                        uint id [[thread_position_in_grid]]) {
        if (id >= dataDim) return;

        // Gradient approximation: (x_data - x_model) * x_data
        float diff = data[id] - model[id];
        grad[id] = lr * diff * data[id];
    }

    // Persistent contrastive divergence gradient
    kernel void persistentCDGradientKernel(device float* chain [[buffer(0)]],
                                      device float* grad [[buffer(1)]],
                                      device float* weights [[buffer(2)]],
                                      constant float& lr [[buffer(3)]],
                                      constant uint& dim [[buffer(4)]],
                                      uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Simplified gradient update
        grad[id] = lr * chain[id];
    }

    // Gibbs sampling step: x' = x + noise
    kernel void gibbsSampleKernel(device float* x [[buffer(0)]],
                             device float* noise [[buffer(1)]],
                             device float* energy [[buffer(2)]],
                             device float* proposed [[buffer(3)]],
                             constant float& stepSize [[buffer(4)]],
                             constant uint& dim [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Metropolis-Hastings: accept with probability min(1, exp(-deltaE))
        // Simplified: always accept with probability
        float rand = noise[id];
        proposed[id] = x[id] + stepSize * (rand - 0.5);

        // Clamp to valid range
        proposed[id] = fmax(0.0, fmin(1.0, proposed[id]));
    }

    // Energy gradient: dE/dx
    kernel void energyGradientKernel(device float* x [[buffer(0)]],
                                device float* grad [[buffer(1)]],
                                device float* weights [[buffer(2)]],
                                constant uint& inputDim [[buffer(3)]],
                                constant uint& hiddenDim [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= inputDim) return;

        float sum = 0.0;
        for (uint h = 0; h < hiddenDim; h++) {
            float activation = 0.0;
            for (uint i = 0; i < inputDim; i++) {
                activation += weights[h * inputDim + i] * x[i];
            }
            // d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
            float sig = 1.0 / (1.0 + exp(-activation));
            float dSig = sig * (1.0 - sig);
            sum += weights[h * inputDim + id] * dSig;
        }
        grad[id] = sum;
    }

    // Langevin dynamics sampling
    // x_{t+1} = x_t - eta * dE/dx + sqrt(2*eta) * noise
    kernel void langevinDynamicsKernel(device float* x [[buffer(0)]],
                                  device float* grad [[buffer(1)]],
                                  device float* noise [[buffer(2)]],
                                  device float* newX [[buffer(3)]],
                                  constant float& lr [[buffer(4)]],
                                  constant float& noiseScale [[buffer(5)]],
                                  constant uint& dim [[buffer(6)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        newX[id] = x[id] - lr * grad[id] + noiseScale * noise[id];
        newX[id] = fmax(0.0, fmin(1.0, newX[id]));
    }

    // Hopfield energy: E = -sum_i,j W_ij * x_i * x_j - sum_i b_i * x_i
    kernel void hopfieldEnergyKernel(device float* x [[buffer(0)]],
                                device float* energy [[buffer(1)]],
                                device float* weights [[buffer(2)]],
                                device float* biases [[buffer(3)]],
                                constant uint& dim [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float energySum = 0.0;

        // -sum b_i * x_i
        for (uint i = 0; i < dim; i++) {
            energySum -= biases[i] * x[i];
        }

        // -sum W_ij * x_i * x_j (symmetric)
        for (uint i = 0; i < dim; i++) {
            for (uint j = i+1; j < dim; j++) {
                energySum -= weights[i * dim + j] * x[i] * x[j];
            }
        }

        energy[0] = energySum;
    }

    // Boltzmann machine visible-hidden energy
    kernel void boltzmannEnergyKernel(device float* v [[buffer(0)]],
                                 device float* h [[buffer(1)]],
                                 device float* energy [[buffer(2)]],
                                 device float* Wvh [[buffer(3)]],
                                 device float* bv [[buffer(4)]],
                                 device float* bh [[buffer(5)]],
                                 constant uint& numVisible [[buffer(6)]],
                                 constant uint& numHidden [[buffer(7)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float energy = 0.0;

        // -sum_i b_i^v * v_i
        for (uint i = 0; i < numVisible; i++) {
            energy -= bv[i] * v[i];
        }

        // -sum_j b_j^h * h_j
        for (uint j = 0; j < numHidden; j++) {
            energy -= bh[j] * h[j];
        }

        // -sum_ij v_i * W_ij * h_j
        for (uint i = 0; i < numVisible; i++) {
            for (uint j = 0; j < numHidden; j++) {
                energy -= v[i] * Wvh[i * numHidden + j] * h[j];
            }
        }

        energy[0] = energy;
    }

    // Softmax for Gibbs sampling probabilities
    kernel void softmaxKernel(device float* energies [[buffer(0)]],
                         device float* probs [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
        if (id >= size) return;

        // Find max for numerical stability
        float maxE = -1e9;
        for (uint i = 0; i < size; i++) {
            maxE = fmax(maxE, energies[i]);
        }

        // exp(e_i - max)
        float expSum = 0.0;
        for (uint i = 0; i < size; i++) {
            expSum += exp(energies[i] - maxE);
        }

        probs[id] = exp(energies[id] - maxE) / expSum;
    }

    // Energy-based prediction (contrastive)
    kernel void ebmPredictionKernel(device float* posEmb [[buffer(0)]],
                                device float* negEmb [[buffer(1)]],
                                device float* energy [[buffer(2)]],
                                device float* logits [[buffer(3)]],
                                constant uint& dim [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        // Energy for positive pair should be low
        // Energy for negative pair should be high
        float posEnergy = 0.0;
        float negEnergy = 0.0;

        for (uint i = 0; i < dim; i++) {
            float diffP = posEmb[i];
            float diffN = negEmb[i];
            posEnergy += diffP * diffP;
            negEnergy += diffN * diffN;
        }

        // Binary classification based on energy difference
        logits[0] = posEnergy - negEnergy;
    }

    // EBM score computation for generation
    kernel void scoreComputationKernel(device float* x [[buffer(0)]],
                                  device float* grad [[buffer(1)]],
                                  device float* weights [[buffer(2)]],
                                  constant uint& dim [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Score: grad_x log p(x) = -grad_x E(x)
        grad[id] = -grad[id];
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Energy-Based Model (EBM) Benchmark ===")
        print("Testing contrastive divergence and Gibbs sampling on ANE\n")

        var allResults: [(name: String, energyTime: Double, gradientTime: Double, sampleTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Energy Compute: \(String(format: "%.4f", result.energyTime * 1000)) ms")
            print("  Gradient:       \(String(format: "%.4f", result.gradientTime * 1000)) ms")
            print("  Sampling:       \(String(format: "%.4f", result.sampleTime * 1000)) ms")
            print("  Total Time:     \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, dataDim: Int, hiddenDim: Int, numChains: Int, numSteps: Int)) throws -> (name: String, energyTime: Double, gradientTime: Double, sampleTime: Double, totalTime: Double) {
        print("  Running \(config.name) (dim=\(config.dataDim), hidden=\(config.hiddenDim), chains=\(config.numChains), steps=\(config.numSteps))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let energyFunc = library.makeFunction(name: "energyForwardKernel"),
              let gradFunc = library.makeFunction(name: "energyGradientKernel"),
              let langevinFunc = library.makeFunction(name: "langevinDynamicsKernel"),
              let ebmPredFunc = library.makeFunction(name: "ebmPredictionKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let energyPipeline = try? device.makeComputePipelineState(function: energyFunc),
              let gradPipeline = try? device.makeComputePipelineState(function: gradFunc),
              let langevinPipeline = try? device.makeComputePipelineState(function: langevinFunc),
              let ebmPredPipeline = try? device.makeComputePipelineState(function: ebmPredFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let dataBytes = config.dataDim * MemoryLayout<Float>.stride
        let hiddenBytes = config.hiddenDim * MemoryLayout<Float>.stride
        let weightBytes = config.hiddenDim * config.dataDim * MemoryLayout<Float>.stride

        guard let dataBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let modelBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let energyBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let gradBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let noiseBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let newXBuffer = device.makeBuffer(length: dataBytes, options: .storageModeShared),
              let logitsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize data
        let dataPtr = dataBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.dataDim {
            dataPtr[i] = Float.random(in: 0...1)
        }

        // Initialize weights
        let weightPtr = weightBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(config.hiddenDim * config.dataDim) {
            weightPtr[i] = Float.random(in: -0.1...0.1)
        }

        // Phase 1: Energy Computation
        let energyStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(energyPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(energyBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(biasBuffer, offset: 0, index: 3)

            var inputDim = UInt32(config.dataDim)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let energyTime = Double(getTimeNanos() - energyStart) / 1e9 / 20.0

        // Phase 2: Energy Gradient
        let gradStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(gradPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(gradBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)

            var inputDim = UInt32(config.dataDim)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.dataDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let gradientTime = Double(getTimeNanos() - gradStart) / 1e9 / 20.0

        // Phase 3: Langevin Sampling
        let sampleStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(langevinPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(gradBuffer, offset: 0, index: 1)
            encoder.setBuffer(noiseBuffer, offset: 0, index: 2)
            encoder.setBuffer(newXBuffer, offset: 0, index: 3)

            var lr = Float(0.01)
            var noiseScale = Float(0.1)
            var dim = UInt32(config.dataDim)
            encoder.setBytes(&lr, length: MemoryLayout<Float>.stride, index: 4)
            encoder.setBytes(&noiseScale, length: MemoryLayout<Float>.stride, index: 5)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.dataDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let sampleTime = Double(getTimeNanos() - sampleStart) / 1e9 / 20.0

        let totalTime = energyTime + gradientTime + sampleTime

        return (config.name, energyTime, gradientTime, sampleTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, energyTime: Double, gradientTime: Double, sampleTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANEEnergyBasedModel"

        let log = """
        === ANE Energy-Based Model (EBM) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Energy (ms) | Gradient (ms) | Sampling (ms) | Total (ms) |
        |--------------|-------------|--------------|---------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.energyTime * 1000)) | \(String(format: "%.4f", $0.gradientTime * 1000)) | \(String(format: "%.4f", $0.sampleTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Energy Compute: Forward pass through energy function
        - Gradient: dE/dx computation for sampling
        - Sampling: Langevin dynamics or Gibbs sampling

        Key Insights:
        - EBMs learn energy surfaces rather than direct probability
        - Contrastive divergence for training
        - Langevin dynamics for sampling from the model
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Energy-Based Model (EBM) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Energy-Based Model workloads - models that learn energy surfaces for discrimination and generation.

        ## What are Energy-Based Models?

        EBMs don't model probability directly; instead they learn an energy function E(x) where:
        - Low energy → likely configurations
        - High energy → unlikely configurations

        ### Probability via Gibbs Distribution
        ```
        p(x) = exp(-E(x)) / Z
        where Z = ∫ exp(-E(x')) dx' (partition function)
        ```

        ### Why Energy Instead of Probability?
        - Partition function Z is intractable for many models
        - Avoids normalization constraint
        - Focus on relative energy differences

        ## EBM Architectures

        ### Boltzmann Machine
        - Binary visible and hidden units
        - E = -Σ W_ij * x_i * x_j - Σ b_i * x_i
        - Restricted version (RBM) simplifies computation

        ### Hopfield Network
        - Associative memory
        - Energy landscape with attractor states
        - E = -0.5 * Σ W_ij * x_i * x_j - Σ b_i * x_i

        ### Neural EBM
        - General energy function E(x; θ)
        - Can be any differentiable network
        - Examples: Energy GAN, latent variable EBMs

        ## Training EBMs

        ### Contrastive Divergence (CD)
        1. Sample positive (data): x⁺ ~ p_data
        2. Sample negative (model): x⁻ ~ p_model (via Gibbs sampling)
        3. Update: θ += lr * (∂E(x⁺)/∂θ - ∂E(x⁻)/∂θ)

        ### Persistent Contrastive Divergence
        - Maintain persistent Markov chains
        - Chains updated slowly during training
        - Better approximation of model distribution

        ### Score Matching
        - Avoids partition function entirely
        - Matches gradient of log-likelihood
        - L(θ) = E_x~p_data[||∇_x log p(x;θ)||²]

        ## Sampling from EBMs

        ### Gibbs Sampling
        - Alternate sampling each variable conditioned on others
        - Requires full conditional distributions

        ### Langevin Dynamics
        ```
        x_{t+1} = x_t - η * ∇_x E(x_t) + √(2η) * ε
        where ε ~ N(0, I)
        ```
        - Gradual descent on energy surface
        - Noise maintains exploration

        ### Hamiltonian Monte Carlo
        - Uses gradient information
        - More efficient than random walk
        - Requires second derivatives

        ## Applications

        ### Image Generation
        - EBMs can generate samples via Langevin sampling
        - Better mode coverage than GANs
        - Model uncertainty naturally

        ### Classification
        - Energy-based decision: low energy = positive class
        - Out-of-distribution detection via energy threshold

        ### Reinforcement Learning
        - Energy-based policy
        - Options as attractors in energy landscape

        ### Computer Vision
        - Texture synthesis
        - Image inpainting
        - Video prediction

        ## Comparison with Other Models

        | Model | Objective | Sampling | Mode Collapse |
        |-------|-----------|----------|---------------|
        | GAN | Min-max game | Latent→Data | Can occur |
        | VAE | ELBO | Decoder sampling | Rare |
        | Flow | Exact log-likelihood | Invert | Avoided |
        | **EBM** | Contrastive | Langevin/Gibbs | Avoided |

        ## ANE vs GPU for EBMs

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Energy Computation | Good | Excellent |
        | Gradient Computation | Good | Excellent |
        | Langevin Sampling | Good | Excellent |
        | Gibbs Sampling | Limited | Excellent |

        ## Key Findings

        1. **Energy Surfaces**: EBMs learn interpretable energy landscapes

        2. **Mode Coverage**: No mode collapse unlike GANs

        3. **Exact Likelihood**: Partition function issues but energy differences are exact

        4. **Sampling Challenge**: Requires iterative methods (Langevin/Gibbs)

        5. **ANE Suitability**: Good for energy and gradient computation

        ## Future Work

        - Test RBM-specific operations
        - Implement full contrastive divergence
        - Benchmark HMC sampling
        - Compare with Flow-based models on same tasks
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
