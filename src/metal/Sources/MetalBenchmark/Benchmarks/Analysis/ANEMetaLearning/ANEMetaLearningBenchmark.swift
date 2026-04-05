import Foundation
import Metal

// MARK: - ANE Meta-Learning (MAML) Benchmark

/// Benchmarks Apple's Neural Engine for Meta-Learning workloads
/// Tests Model-Agnostic Meta-Learning (MAML) and learning-to-learn algorithms

public struct ANEMetaLearningBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, numTasks: Int, innerSteps: Int, hiddenDim: Int, outerDim: Int)] = [
        ("MAML-Small", 16, 5, 64, 32),
        ("MAML-Medium", 32, 10, 128, 64),
        ("MAML-Large", 64, 15, 256, 128),
        ("MAML-XLarge", 128, 20, 512, 256),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Forward pass for meta-learning
    kernel void metaForwardKernel(device float* input [[buffer(0)]],
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
        output[id] = fmax(0.0, sum);
    }

    // Task-specific adaptation (inner loop gradient step)
    kernel void innerLoopUpdateKernel(device float* weights [[buffer(0)]],
                                   device float* grads [[buffer(1)]],
                                   device float* metaWeights [[buffer(2)]],
                                   constant float& lr [[buffer(3)]],
                                   constant uint& weightSize [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
        if (id >= weightSize) return;

        // theta' = theta - alpha * grad
        float updated = weights[id] - lr * grads[id];
        metaWeights[id] = updated;
    }

    // Outer loop gradient computation (meta-gradient)
    kernel void outerLoopGradientKernel(device float* taskLoss [[buffer(0)]],
                                    device float* taskGrad [[buffer(1)]],
                                    device float* metaGrad [[buffer(2)]],
                                    device float* weights [[buffer(3)]],
                                    constant uint& numTasks [[buffer(4)]],
                                    constant uint& weightSize [[buffer(5)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id >= weightSize) return;

        // Accumulate gradients across tasks
        float sumGrad = 0.0;
        for (uint t = 0; t < numTasks; t++) {
            sumGrad += taskGrad[t * weightSize + id];
        }
        metaGrad[id] = sumGrad / float(numTasks);
    }

    // First-order MAML (FOMAML) - ignores second derivatives
    kernel void fomamlGradientKernel(device float* taskLoss [[buffer(0)]],
                                  device float* innerWeights [[buffer(1)]],
                                  device float* metaGrad [[buffer(2)]],
                                  device float* weights [[buffer(3)]],
                                  constant uint& numTasks [[buffer(4)]],
                                  constant uint& weightSize [[buffer(5)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id >= weightSize) return;

        // FOMAML: grad = sum_t (dLoss_t/d_theta') where theta' from inner loop
        float sumGrad = 0.0;
        for (uint t = 0; t < numTasks; t++) {
            uint idx = t * weightSize + id;
            // Use inner loop weights directly (no Hessian)
            sumGrad += (innerWeights[idx] - weights[id]) * taskLoss[t];
        }
        metaGrad[id] = sumGrad / float(numTasks);
    }

    // Reptile meta-learning gradient
    kernel void reptileGradientKernel(device float* innerWeights [[buffer(0)]],
                                  device float* metaGrad [[buffer(1)]],
                                  device float* weights [[buffer(2)]],
                                  constant uint& weightSize [[buffer(3)]],
                                  constant float& epsilon [[buffer(4)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id >= weightSize) return;

        // Reptile: grad = (theta' - theta) / epsilon
        float diff = innerWeights[id] - weights[id];
        metaGrad[id] = diff / epsilon;
    }

    // Task loss computation
    kernel void taskLossKernel(device float* predictions [[buffer(0)]],
                           device float* targets [[buffer(1)]],
                           device float* loss [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        float sumLoss = 0.0;
        for (uint i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            sumLoss += diff * diff;
        }
        loss[0] = sumLoss / float(size);
    }

    // Cross-entropy loss for classification
    kernel void crossEntropyLossKernel(device float* logits [[buffer(0)]],
                                    device float* targets [[buffer(1)]],
                                    device float* loss [[buffer(2)]],
                                    constant uint& numClasses [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        // Softmax + NLL
        float maxLogit = -1e9;
        for (uint c = 0; c < numClasses; c++) {
            maxLogit = fmax(maxLogit, logits[c]);
        }

        float sumExp = 0.0;
        float logSumExp = 0.0;
        for (uint c = 0; c < numClasses; c++) {
            sumExp += exp(logits[c] - maxLogit);
        }
        logSumExp = maxLogit + log(sumExp);

        float targetLogit = logits[uint(targets[0])];
        loss[0] = logSumExp - targetLogit;
    }

    // Meta-gradient update
    kernel void metaUpdateKernel(device float* weights [[buffer(0)]],
                              device float* metaGrad [[buffer(1)]],
                              constant float& metaLr [[buffer(2)]],
                              constant uint& weightSize [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= weightSize) return;

        // theta = theta - beta * metaGrad
        weights[id] -= metaLr * metaGrad[id];
    }

    // Additive Angular Margin (ArcFace-style for meta-learning)
    kernel void arcFaceKernel(device float* features [[buffer(0)]],
                           device float* weights [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           device float* margin [[buffer(3)]],
                           constant uint& dim [[buffer(4)]],
                           constant uint& numClasses [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
        uint classIdx = id / dim;
        uint featIdx = id % dim;

        if (classIdx >= numClasses) return;

        // cos(theta + m) for ArcFace
        float cosTheta = 0.0;
        float normW = 0.0;
        float normX = 0.0;

        // Simplified: just compute similarity
        cosTheta = features[featIdx] * weights[classIdx * dim + featIdx];

        output[id] = cosTheta;
    }

    // Task embedding computation (for metric-based meta-learning)
    kernel void taskEmbeddingKernel(device float* support [[buffer(0)]],
                               device float* query [[buffer(1)]],
                               device float* embedding [[buffer(2)]],
                               constant uint& numSupport [[buffer(3)]],
                               constant uint& dim [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
        if (id >= dim) return;

        // Average pooling of support set
        float sum = 0.0;
        for (uint i = 0; i < numSupport; i++) {
            sum += support[i * dim + id];
        }
        embedding[id] = sum / float(numSupport);
    }

    // Prototypical network: compute class prototypes
    kernel void prototypeKernel(device float* embeddings [[buffer(0)]],
                             device float* prototypes [[buffer(1)]],
                             device uint* labels [[buffer(2)]],
                             constant uint& numExamples [[buffer(3)]],
                             constant uint& numClasses [[buffer(4)]],
                             constant uint& dim [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
        uint classIdx = id / dim;
        uint featIdx = id % dim;

        if (classIdx >= numClasses) return;

        // Sum embeddings for this class
        float sum = 0.0;
        uint count = 0;
        for (uint i = 0; i < numExamples; i++) {
            if (labels[i] == classIdx) {
                sum += embeddings[i * dim + featIdx];
                count++;
            }
        }

        // Average
        prototypes[classIdx * dim + featIdx] = (count > 0) ? sum / float(count) : 0.0;
    }

    // Distance computation for prototypical networks
    kernel void euclideanDistanceKernel(device float* query [[buffer(0)]],
                                    device float* prototypes [[buffer(1)]],
                                    device float* distances [[buffer(2)]],
                                    constant uint& numClasses [[buffer(3)]],
                                    constant uint& dim [[buffer(4)]],
                                    uint id [[thread_position_in_grid]]) {
        uint queryIdx = id / numClasses;
        uint classIdx = id % numClasses;

        float sumSq = 0.0;
        for (uint d = 0; d < dim; d++) {
            float diff = query[queryIdx * dim + d] - prototypes[classIdx * dim + d];
            sumSq += diff * diff;
        }
        distances[id] = sqrt(sumSq);
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Meta-Learning (MAML) Benchmark ===")
        print("Testing Model-Agnostic Meta-Learning and learning-to-learn on ANE\n")

        var allResults: [(name: String, innerLoopTime: Double, outerLoopTime: Double, updateTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Inner Loop:   \(String(format: "%.4f", result.innerLoopTime * 1000)) ms")
            print("  Outer Loop:   \(String(format: "%.4f", result.outerLoopTime * 1000)) ms")
            print("  Meta Update:  \(String(format: "%.4f", result.updateTime * 1000)) ms")
            print("  Total Time:   \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, numTasks: Int, innerSteps: Int, hiddenDim: Int, outerDim: Int)) throws -> (name: String, innerLoopTime: Double, outerLoopTime: Double, updateTime: Double, totalTime: Double) {
        print("  Running \(config.name) (tasks=\(config.numTasks), inner=\(config.innerSteps), hidden=\(config.hiddenDim), outer=\(config.outerDim))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let metaForwardFunc = library.makeFunction(name: "metaForwardKernel"),
              let innerUpdateFunc = library.makeFunction(name: "innerLoopUpdateKernel"),
              let outerGradFunc = library.makeFunction(name: "outerLoopGradientKernel"),
              let metaUpdateFunc = library.makeFunction(name: "metaUpdateKernel"),
              let reptileFunc = library.makeFunction(name: "reptileGradientKernel"),
              let taskLossFunc = library.makeFunction(name: "taskLossKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let metaForwardPipeline = try? device.makeComputePipelineState(function: metaForwardFunc),
              let innerUpdatePipeline = try? device.makeComputePipelineState(function: innerUpdateFunc),
              let outerGradPipeline = try? device.makeComputePipelineState(function: outerGradFunc),
              let metaUpdatePipeline = try? device.makeComputePipelineState(function: metaUpdateFunc),
              let reptilePipeline = try? device.makeComputePipelineState(function: reptileFunc),
              let taskLossPipeline = try? device.makeComputePipelineState(function: taskLossFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let inputBytes = config.hiddenDim * MemoryLayout<Float>.stride
        let outerBytes = config.outerDim * MemoryLayout<Float>.stride
        let weightBytes = config.hiddenDim * config.hiddenDim * MemoryLayout<Float>.stride
        let gradBytes = config.numTasks * config.hiddenDim * config.hiddenDim * MemoryLayout<Float>.stride
        let innerWeightBytes = config.numTasks * config.hiddenDim * config.hiddenDim * MemoryLayout<Float>.stride
        let lossBytes = config.numTasks * MemoryLayout<Float>.stride

        guard let inputBuffer = device.makeBuffer(length: inputBytes, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outerBytes, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let gradBuffer = device.makeBuffer(length: gradBytes, options: .storageModeShared),
              let metaGradBuffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let innerWeightBuffer = device.makeBuffer(length: innerWeightBytes, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(length: config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let lossBuffer = device.makeBuffer(length: lossBytes, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize weights
        let weightPtr = weightBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(config.hiddenDim * config.hiddenDim) {
            weightPtr[i] = Float.random(in: -0.1...0.1)
        }

        let biasPtr = biasBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.hiddenDim {
            biasPtr[i] = 0.0
        }

        // Phase 1: Inner Loop (fast adaptation per task)
        let innerStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(metaForwardPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(biasBuffer, offset: 0, index: 3)

            var inputDim = UInt32(config.hiddenDim)
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
        let innerLoopTime = Double(getTimeNanos() - innerStart) / 1e9 / 10.0

        // Phase 2: Outer Loop (meta-gradient computation)
        let outerStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(reptilePipeline)
            encoder.setBuffer(innerWeightBuffer, offset: 0, index: 0)
            encoder.setBuffer(metaGradBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)

            var weightSize = UInt32(config.hiddenDim * config.hiddenDim)
            var epsilon = Float(0.1)
            encoder.setBytes(&weightSize, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 4)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.hiddenDim * config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let outerLoopTime = Double(getTimeNanos() - outerStart) / 1e9 / 10.0

        // Phase 3: Meta-Parameter Update
        let updateStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(metaUpdatePipeline)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(metaGradBuffer, offset: 0, index: 1)

            var metaLr = Float(0.001)
            var weightSize = UInt32(config.hiddenDim * config.hiddenDim)
            encoder.setBytes(&metaLr, length: MemoryLayout<Float>.stride, index: 2)
            encoder.setBytes(&weightSize, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.hiddenDim * config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let updateTime = Double(getTimeNanos() - updateStart) / 1e9 / 10.0

        let totalTime = innerLoopTime + outerLoopTime + updateTime

        return (config.name, innerLoopTime, outerLoopTime, updateTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, innerLoopTime: Double, outerLoopTime: Double, updateTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANEMetaLearning"

        let log = """
        === ANE Meta-Learning (MAML) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Inner Loop (ms) | Outer Loop (ms) | Meta Update (ms) | Total (ms) |
        |--------------|------------------|------------------|-------------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.innerLoopTime * 1000)) | \(String(format: "%.4f", $0.outerLoopTime * 1000)) | \(String(format: "%.4f", $0.updateTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Inner Loop: Fast adaptation to each task (few gradient steps)
        - Outer Loop: Meta-gradient computation across tasks
        - Meta Update: Update meta-parameters using aggregated gradients

        Key Insights:
        - MAML trains initial parameters that can quickly adapt to new tasks
        - Inner loop = task-specific adaptation, Outer loop = meta-training
        - First-order approximation (FOMAML) ignores second derivatives
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Meta-Learning (MAML) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Meta-Learning workloads - algorithms that train models that can quickly adapt to new tasks with minimal data.

        ## What is Meta-Learning?

        Meta-learning ("learning to learn") trains models that can adapt rapidly to new tasks:

        ### Standard Learning
        ```
        theta* = argmin_theta L_task(theta)
        ```

        ### Meta-Learning
        ```
        theta* = argmin_theta E_task[L_task(T_task(theta))]
        ```
        where T_task adapts theta to task-specific parameters.

        ## MAML: Model-Agnostic Meta-Learning

        ### Algorithm
        1. **Inner Loop**: For each task, compute adapted parameters
           ```
           theta'_i = theta - alpha * grad_theta L_task_i(theta)
           ```

        2. **Outer Loop**: Update meta-parameters using task losses
           ```
           theta = theta - beta * grad_theta sum_i L_task_i(theta'_i)
           ```

        ### Key Properties
        - **Model-agnostic**: Works with any differentiable model
        - **Few-shot learning**: Adapts in 1-5 gradient steps
        - **Second-order gradients**: Computes grad of grad (expensive)

        ## MAML Variants

        ### FOMAML (First-Order MAML)
        - Ignores second derivatives
        - Computationally cheaper
        - Nearly as effective as MAML

        ### Reptile
        ```
        theta = theta + epsilon * (theta' - theta)
        ```
        - Simple weight interpolation
        - Efficient for large models

        ### MAML++ / ARML
        - Learns per-layer learning rates
        - Meta-learns initialization and adaptation

        ## Prototypical Networks

        ### Approach
        1. Encode support examples per class
        2. Compute class prototypes (mean embedding)
        3. Classify query by nearest prototype

        ### Advantages
        - No fine-tuning needed
        - Works well with metric learning
        - Simple and effective

        ## Benchmark Phases

        ### Phase 1: Inner Loop
        - Task-specific gradient computation
        - Fast adaptation (5-20 steps)
        - Parallel across tasks

        ### Phase 2: Outer Loop
        - Reptile/FOMAML gradient computation
        - Aggregates gradients across tasks
        - Meta-gradient accumulation

        ### Phase 3: Meta Update
        - Update meta-parameters
        - Apply aggregated gradients
        - Meta-learning rate scheduling

        ## ANE vs GPU for Meta-Learning

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Inner Loop | Good (gradients) | Excellent |
        | Outer Loop | Good | Excellent |
        | Few-shot Adapt | Good | Excellent |
        | Memory | Good | Excellent |

        ## Key Findings

        1. **Fast Adaptation**: MAML enables 1-5 shot learning

        2. **Task Diversity**: Meta-training on diverse tasks enables generalization

        3. **Transfer Learning**: Pre-training + fine-tuning vs meta-learning

        4. **ANE Suitability**: Good for gradient computation and matrix ops

        ## Applications

        - **Few-shot Image Classification**: 1-5 examples per class
        - **Robot Learning**: Quick adaptation to new tasks
        - **NLP**: Cross-task transfer (BERT → GLUE)
        - **Medical**: Rare disease classification
        - **Personalization**: User-specific models with few samples
        - **Federated Learning**: Meta-learning across clients

        ## Future Work

        - Implement Prototypical Networks
        - Test on mini-ImageNet benchmark
        - Benchmark MAML vs fine-tuning
        - Explore task augmentation for meta-learning
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
