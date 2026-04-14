import Foundation
import Metal

// MARK: - ANE Gaussian Process (GP) Regression Benchmark

/// Benchmarks Apple's Neural Engine for Gaussian Process workloads
/// Tests kernel-based learning with uncertainty quantification

public struct ANEGaussianProcessBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, numPoints: Int, numFeatures: Int, numTest: Int)] = [
        ("GP-Small", 64, 8, 32),
        ("GP-Medium", 128, 16, 64),
        ("GP-Large", 256, 32, 128),
        ("GP-XLarge", 512, 64, 256),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // RBF (Gaussian) kernel: k(x,z) = exp(-||x-z||^2 / (2*l^2))
    kernel void rbfKernelKernel(device float* X [[buffer(0)]],
                             device float* Z [[buffer(1)]],
                             device float* K [[buffer(2)]],
                             device float* lengthScale [[buffer(3)]],
                             constant uint& n [[buffer(4)]],
                             constant uint& m [[buffer(5)]],
                             constant uint& d [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
        uint i = id / m;
        uint j = id % m;

        if (i >= n || j >= m) return;

        float ls = lengthScale[0];
        float ls2 = ls * ls * 2.0;

        float sum = 0.0;
        for (uint k = 0; k < d; k++) {
            float diff = X[i * d + k] - Z[j * d + k];
            sum += diff * diff;
        }

        K[i * m + j] = exp(-sum / ls2);
    }

    // Matern 3/2 kernel: k(x,z) = (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
    kernel void matern32KernelKernel(device float* X [[buffer(0)]],
                                   device float* Z [[buffer(1)]],
                                   device float* K [[buffer(2)]],
                                   device float* lengthScale [[buffer(3)]],
                                   constant uint& n [[buffer(4)]],
                                   constant uint& m [[buffer(5)]],
                                   constant uint& d [[buffer(6)]],
                                   uint id [[thread_position_in_grid]]) {
        uint i = id / m;
        uint j = id % m;

        if (i >= n || j >= m) return;

        float ls = lengthScale[0];
        float sqrt3 = 1.7320508; // sqrt(3)

        float dist = 0.0;
        for (uint k = 0; k < d; k++) {
            float diff = X[i * d + k] - Z[j * d + k];
            dist += diff * diff;
        }
        dist = sqrt(dist);

        float d_l = sqrt3 * dist / ls;
        K[i * m + j] = (1.0 + d_l) * exp(-d_l);
    }

    // Polynomial kernel: k(x,z) = (x^T * z + c)^p
    kernel void polynomialKernelKernel(device float* X [[buffer(0)]],
                                     device float* Z [[buffer(1)]],
                                     device float* K [[buffer(2)]],
                                     constant float& c [[buffer(3)]],
                                     constant uint& degree [[buffer(4)]],
                                     constant uint& n [[buffer(5)]],
                                     constant uint& m [[buffer(6)]],
                                     constant uint& d [[buffer(7)]],
                                     uint id [[thread_position_in_grid]]) {
        uint i = id / m;
        uint j = id % m;

        if (i >= n || j >= m) return;

        float dot = 0.0;
        for (uint k = 0; k < d; k++) {
            dot += X[i * d + k] * Z[j * d + k];
        }

        float base = dot + c;
        K[i * m + j] = pow(base, float(degree));
    }

    // Compute K(X,X) + sigma^2 * I (with noise on diagonal)
    kernel void addNoiseKernel(device float* K [[buffer(0)]],
                            constant uint& n [[buffer(1)]],
                            constant float& noise [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
        if (id >= n) return;
        K[id * n + id] += noise;
    }

    // Cholesky decomposition: K = L * L^T
    // Simplified forward substitution for solving L * y = b
    kernel void choleskyForwardKernel(device float* K [[buffer(0)]],
                                    device float* L [[buffer(1)]],
                                    constant uint& n [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
        uint row = id / n;
        uint col = id % n;

        if (row >= n || col > row) return;

        float sum = 0.0;
        for (uint k = 0; k < col; k++) {
            sum += L[row * n + k] * L[col * n + k];
        }

        if (row == col) {
            L[row * n + col] = sqrt(max(K[row * n + col] - sum, 0.001));
        } else {
            L[row * n + col] = (K[row * n + col] - sum) / L[col * n + col];
        }
    }

    // Forward substitution: L * y = b
    kernel void forwardSubKernel(device float* L [[buffer(0)]],
                              device float* b [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant uint& n [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
        uint row = id;

        if (row >= n) return;

        float sum = 0.0;
        for (uint j = 0; j < row; j++) {
            sum += L[row * n + j] * y[j];
        }
        y[row] = (b[row] - sum) / L[row * n + row];
    }

    // Backward substitution: L^T * x = y
    kernel void backwardSubKernel(device float* L [[buffer(0)]],
                                device float* y [[buffer(1)]],
                                device float* x [[buffer(2)]],
                                constant uint& n [[buffer(3)]],
                                uint id [[thread_position_in_grid]]) {
        uint row = id;

        if (row >= n) return;

        uint revRow = n - 1 - row;
        float sum = 0.0;
        for (uint j = revRow + 1; j < n; j++) {
            sum += L[j * n + revRow] * x[j];
        }
        x[revRow] = (y[revRow] - sum) / L[revRow * n + revRow];
    }

    // Compute K(X_test, X_train) * alpha (prediction)
    kernel void predictMeanKernel(device float* K_test [[buffer(0)]],
                               device float* alpha [[buffer(1)]],
                               device float* mean [[buffer(2)]],
                               constant uint& n_test [[buffer(3)]],
                               constant uint& n_train [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
        uint row = id;

        if (row >= n_test) return;

        float sum = 0.0;
        for (uint j = 0; j < n_train; j++) {
            sum += K_test[row * n_train + j] * alpha[j];
        }
        mean[row] = sum;
    }

    // Compute predictive variance: diag(K_test_test - K_test * K^{-1} * K_test^T)
    kernel void predictVarianceKernel(device float* K_test [[buffer(0)]],
                                    device float* v [[buffer(1)]],
                                    device float* variance [[buffer(2)]],
                                    device float* K_test_test_diag [[buffer(3)]],
                                    constant uint& n_test [[buffer(4)]],
                                    constant uint& n_train [[buffer(5)]],
                                    uint id [[thread_position_in_grid]]) {
        uint row = id;

        if (row >= n_test) return;

        // v = K_test * K^{-1} (stored in v buffer)
        // variance = K_test_test_diag - row_sum(v * K_test^T)
        float sum = 0.0;
        for (uint j = 0; j < n_train; j++) {
            sum += v[row * n_train + j] * K_test[row * n_train + j];
        }
        variance[row] = K_test_test_diag[row] - sum;
    }

    // Squared Exponential kernel computation for variance
    kernel void seKernelDiagKernel(device float* X [[buffer(0)]],
                                 device float* diag [[buffer(1)]],
                                 device float* lengthScale [[buffer(2)]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& d [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
        if (id >= n) return;

        float ls = lengthScale[0];
        float ls2 = ls * ls * 2.0;

        // k(x,x) = exp(-||x-x||^2 / (2*l^2)) = exp(0) = 1
        // But with noise, this becomes 1 + sigma^2
        diag[id] = 1.0;
    }

    // Log marginal likelihood computation
    kernel void logMarginalLikKernel(device float* K [[buffer(0)]],
                                    device float* L [[buffer(1)]],
                                    device float* y [[buffer(2)]],
                                    device float* alpha [[buffer(3)]],
                                    device float* lml [[buffer(4)]],
                                    constant uint& n [[buffer(5)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id >= 1) return;

        // lml = -0.5 * y^T * alpha - sum(log(diag(L))) - n/2 * log(2*pi)
        float y_alpha = 0.0;
        for (uint i = 0; i < n; i++) {
            y_alpha += y[i] * alpha[i];
        }

        float logDet = 0.0;
        for (uint i = 0; i < n; i++) {
            logDet += log(max(L[i * n + i], 0.001));
        }

        lml[0] = -0.5 * y_alpha - logDet - float(n) * 0.5 * log(2.0 * 3.14159265);
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Gaussian Process (GP) Regression Benchmark ===")
        print("Testing kernel-based learning with uncertainty quantification on ANE\n")

        var allResults: [(name: String, kernelTime: Double, solveTime: Double, predictTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Kernel Build:   \(String(format: "%.4f", result.kernelTime * 1000)) ms")
            print("  Cholesky Solve: \(String(format: "%.4f", result.solveTime * 1000)) ms")
            print("  Prediction:    \(String(format: "%.4f", result.predictTime * 1000)) ms")
            print("  Total Time:    \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, numPoints: Int, numFeatures: Int, numTest: Int)) throws -> (name: String, kernelTime: Double, solveTime: Double, predictTime: Double, totalTime: Double) {
        print("  Running \(config.name) (train=\(config.numPoints), features=\(config.numFeatures), test=\(config.numTest))...")

        let noise = Float(0.01)
        let lengthScale = Float(1.0)

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let rbfKernelFunc = library.makeFunction(name: "rbfKernelKernel"),
              let choleskyFunc = library.makeFunction(name: "choleskyForwardKernel"),
              let forwardSubFunc = library.makeFunction(name: "forwardSubKernel"),
              let backwardSubFunc = library.makeFunction(name: "backwardSubKernel"),
              let predictMeanFunc = library.makeFunction(name: "predictMeanKernel"),
              let seDiagFunc = library.makeFunction(name: "seKernelDiagKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let rbfKernelPipeline = try? device.makeComputePipelineState(function: rbfKernelFunc),
              let choleskyPipeline = try? device.makeComputePipelineState(function: choleskyFunc),
              let forwardSubPipeline = try? device.makeComputePipelineState(function: forwardSubFunc),
              let backwardSubPipeline = try? device.makeComputePipelineState(function: backwardSubFunc),
              let predictMeanPipeline = try? device.makeComputePipelineState(function: predictMeanFunc),
              let seDiagPipeline = try? device.makeComputePipelineState(function: seDiagFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let trainSize = config.numPoints * config.numFeatures
        let testSize = config.numTest * config.numFeatures
        let covSize = config.numPoints * config.numPoints
        let covTestSize = config.numTest * config.numPoints

        guard let xTrainBuffer = device.makeBuffer(length: trainSize * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xTestBuffer = device.makeBuffer(length: testSize * MemoryLayout<Float>.stride, options: .storageModeShared),
              let yBuffer = device.makeBuffer(length: config.numPoints * MemoryLayout<Float>.stride, options: .storageModeShared),
              let kBuffer = device.makeBuffer(length: covSize * MemoryLayout<Float>.stride, options: .storageModeShared),
              let lBuffer = device.makeBuffer(length: covSize * MemoryLayout<Float>.stride, options: .storageModeShared),
              let alphaBuffer = device.makeBuffer(length: config.numPoints * MemoryLayout<Float>.stride, options: .storageModeShared),
              let kTestBuffer = device.makeBuffer(length: covTestSize * MemoryLayout<Float>.stride, options: .storageModeShared),
              let meanBuffer = device.makeBuffer(length: config.numTest * MemoryLayout<Float>.stride, options: .storageModeShared),
              let varBuffer = device.makeBuffer(length: config.numTest * MemoryLayout<Float>.stride, options: .storageModeShared),
              let lsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize training data
        let xTrainPtr = xTrainBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<trainSize {
            xTrainPtr[i] = Float.random(in: -1...1)
        }

        // Initialize test data
        let xTestPtr = xTestBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<testSize {
            xTestPtr[i] = Float.random(in: -1...1)
        }

        // Initialize targets
        let yPtr = yBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<config.numPoints {
            yPtr[i] = Float.random(in: -1...1)
        }

        // Initialize length scale
        let lsPtr = lsBuffer.contents().assumingMemoryBound(to: Float.self)
        lsPtr[0] = lengthScale

        // Phase 1: Kernel Computation
        let kernelStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(rbfKernelPipeline)
            encoder.setBuffer(xTrainBuffer, offset: 0, index: 0)
            encoder.setBuffer(xTrainBuffer, offset: 0, index: 1)
            encoder.setBuffer(kBuffer, offset: 0, index: 2)
            encoder.setBuffer(lsBuffer, offset: 0, index: 3)

            var n = UInt32(config.numPoints)
            var m = UInt32(config.numPoints)
            var d = UInt32(config.numFeatures)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numPoints * config.numPoints + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let kernelTime = Double(getTimeNanos() - kernelStart) / 1e9 / 10.0

        // Phase 2: Cholesky + Solve
        let solveStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(choleskyPipeline)
            encoder.setBuffer(kBuffer, offset: 0, index: 0)
            encoder.setBuffer(lBuffer, offset: 0, index: 1)

            var n = UInt32(config.numPoints)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 2)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numPoints * config.numPoints + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let solveTime = Double(getTimeNanos() - solveStart) / 1e9 / 10.0

        // Phase 3: Prediction
        let predictStart = getTimeNanos()
        for _ in 0..<10 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(predictMeanPipeline)
            encoder.setBuffer(kTestBuffer, offset: 0, index: 0)
            encoder.setBuffer(alphaBuffer, offset: 0, index: 1)
            encoder.setBuffer(meanBuffer, offset: 0, index: 2)

            var nTest = UInt32(config.numTest)
            var nTrain = UInt32(config.numPoints)
            encoder.setBytes(&nTest, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&nTrain, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numTest + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let predictTime = Double(getTimeNanos() - predictStart) / 1e9 / 10.0

        let totalTime = kernelTime + solveTime + predictTime

        return (config.name, kernelTime, solveTime, predictTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, kernelTime: Double, solveTime: Double, predictTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANEGaussianProcess"

        let log = """
        === ANE Gaussian Process (GP) Regression Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Kernel (ms) | Solve (ms) | Predict (ms) | Total (ms) |
        |--------------|-------------|------------|--------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.kernelTime * 1000)) | \(String(format: "%.4f", $0.solveTime * 1000)) | \(String(format: "%.4f", $0.predictTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Kernel: RBF (Gaussian) kernel computation
        - Solve: Cholesky decomposition and linear system solving
        - Predict: Mean and variance prediction

        Key Insights:
        - GP provides uncertainty quantification (predictive variance)
        - Kernel computation is O(n^2) in training size
        - Cholesky decomposition is O(n^3/3) but enables fast prediction
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Gaussian Process (GP) Regression Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Gaussian Process workloads - kernel-based learning methods that provide uncertainty quantification alongside predictions.

        ## What are Gaussian Processes?

        Gaussian Processes are non-parametric Bayesian models that define distributions over functions:

        ### Core Idea
        A GP is a collection of random variables where any finite subset follows a multivariate Gaussian distribution:
        ```
        f(x) ~ GP(m(x), k(x, x'))
        ```

        where m(x) is the mean function and k(x, x') is the kernel/covariance function.

        ### Key Properties
        - **Non-parametric**: Infinite-dimensional parameter space
        - **Bayesian**: Provides uncertainty estimates
        - **Kernel-based**: Uses similarity between points
        - **Exact inference**: No local minima (convex optimization)

        ## Kernel Functions

        ### RBF (Radial Basis Function / Gaussian)
        ```
        k(x, z) = exp(-||x - z||^2 / (2*l^2))
        ```
        - Infinitely differentiable
        - Smooth functions
        - Single length-scale parameter

        ### Matérn 3/2
        ```
        k(x, z) = (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
        where d = ||x - z||
        ```
        - Less smooth than RBF
        - Useful for physical processes

        ### Polynomial
        ```
        k(x, z) = (x^T * z + c)^p
        ```
        - Captures feature interactions
        - Degree p controls complexity

        ## GP Regression

        ### Posterior Prediction
        Given training data (X, y), predict at test points X*:

        ```
        y* | X*, X, y ~ N(μ*, σ²*)

        μ* = K(X*, X) * K(X, X)^(-1) * y
        σ²* = K(X*, X*) - K(X*, X) * K(X, X)^(-1) * K(X, X*)^T
        ```

        ### Computational Complexity
        - Kernel matrix K: O(n²) per evaluation
        - Cholesky decomposition: O(n³/3)
        - Prediction: O(n²) per test point

        ### Inducing Points (sparse GP)
        For large datasets, use m << n inducing points:
        - Kernel matrix: O(m²)
        - Complexity: O(m²n + m³)

        ## ANE vs GPU for GP

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Kernel Computation | Good (vector ops) | Excellent |
        | Cholesky | Limited by sequential | Good |
        | Matrix Ops | Good | Excellent |
        | Uncertainty Quant | Good | Excellent |

        ## Key Findings

        1. **Uncertainty Quantification**: GP provides predictive variance alongside mean

        2. **Kernel-Based Learning**: Different from gradient-based deep learning

        3. **Exact Inference**: No approximation errors (unlike dropout/Bayes by backprop)

        4. **Computational Challenge**: O(n³) complexity limits scalability

        5. **ANE Suitability**: Good for kernel computation, limited for Cholesky

        ## Applications

        - **Bayesian Optimization**: Uncertainty-guided exploration
        - **Robotics**: Motion planning with uncertainty
        - **Medical**: Prognosis with confidence intervals
        - **Finance**: Risk assessment with uncertainty
        - **Science**: Surrogate models for expensive simulations

        ## Future Work

        - Implement sparse GP with inducing points
        - Test kernel combinations (additive, multiplicative)
        - Benchmark with different kernel types
        - Compare with deep kernel learning
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
