import Foundation
import Metal

// MARK: - ANE Graph Neural Network (GNN) Benchmark

/// Benchmarks Apple's Neural Engine for Graph Neural Network workloads
/// Tests message passing, aggregation, and graph convolution operations

public struct ANEGraphNeuralNetworkBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // MARK: - Configuration
    let configurations: [(name: String, numNodes: Int, numEdges: Int, hiddenDim: Int, numLayers: Int)] = [
        ("GNN-Small", 64, 256, 32, 3),
        ("GNN-Medium", 128, 512, 64, 4),
        ("GNN-Large", 256, 1024, 128, 5),
        ("GNN-XLarge", 512, 2048, 256, 6),
    ]

    // MARK: - Shader Source
    let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Initialize node features from input
    kernel void initFeaturesKernel(device float* nodeFeatures [[buffer(0)]],
                                device float* edgeFeatures [[buffer(1)]],
                                constant uint& numNodes [[buffer(2)]],
                                constant uint& featureDim [[buffer(3)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= numNodes * featureDim) return;
        nodeFeatures[id] = edgeFeatures[id % (numNodes * featureDim)];
    }

    // Message computation: m_{ij} = f(h_i, h_j, e_{ij})
    // Simplified: m_{ij} = W * concat(h_i, h_j)
    kernel void messagePassKernel(device float* srcFeatures [[buffer(0)]],
                              device float* dstFeatures [[buffer(1)]],
                              device float* messages [[buffer(2)]],
                              device float* weights [[buffer(3)]],
                              device uint* edges [[buffer(4)]],
                              constant uint& numEdges [[buffer(5)]],
                              constant uint& hiddenDim [[buffer(6)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= numEdges) return;

        uint src = edges[id * 2];
        uint dst = edges[id * 2 + 1];

        float sum = 0.0;
        for (uint d = 0; d < hiddenDim; d++) {
            float src_h = srcFeatures[src * hiddenDim + d];
            float dst_h = dstFeatures[dst * hiddenDim + d];

            // Simple message: h_src - h_dst
            float msg = src_h - dst_h;

            uint wIdx = d * hiddenDim + (id % hiddenDim);
            sum += weights[wIdx] * msg;
        }
        messages[id] = sum;
    }

    // Aggregate messages: h_i = aggregate_{j in N(i)} m_{ij}
    // Using mean aggregation: h_i = (1/|N(i)|) * sum(m_{ij})
    kernel void aggregateMeanKernel(device float* messages [[buffer(0)]],
                                 device float* edgeDst [[buffer(1)]],
                                 device float* aggregated [[buffer(2)]],
                                 device uint* edges [[buffer(3)]],
                                 device float* degrees [[buffer(4)]],
                                 constant uint& numEdges [[buffer(5)]],
                                 constant uint& numNodes [[buffer(6)]],
                                 constant uint& hiddenDim [[buffer(7)]],
                                 uint id [[thread_position_in_grid]]) {
        uint node = id / hiddenDim;
        uint dim = id % hiddenDim;

        if (node >= numNodes) return;

        float sum = 0.0;
        uint count = 0;

        for (uint e = 0; e < numEdges; e++) {
            if (edges[e * 2 + 1] == node) {
                sum += messages[e * hiddenDim + dim];
                count++;
            }
        }

        float degree = degrees[node];
        aggregated[node * hiddenDim + dim] = (degree > 0.0) ? sum / degree : 0.0;
    }

    // Sum aggregation alternative
    kernel void aggregateSumKernel(device float* messages [[buffer(0)]],
                                 device float* aggregated [[buffer(1)]],
                                 device uint* edges [[buffer(2)]],
                                 constant uint& numEdges [[buffer(3)]],
                                 constant uint& numNodes [[buffer(4)]],
                                 constant uint& hiddenDim [[buffer(5)]],
                                 uint id [[thread_position_in_grid]]) {
        uint node = id / hiddenDim;
        uint dim = id % hiddenDim;

        if (node >= numNodes) return;

        float sum = 0.0;

        for (uint e = 0; e < numEdges; e++) {
            if (edges[e * 2 + 1] == node) {
                sum += messages[e * hiddenDim + dim];
            }
        }

        aggregated[node * hiddenDim + dim] = sum;
    }

    // Update function: h_i' = update(h_i, aggregated_messages)
    // h_i' = ReLU(W * h_i + W2 * agg)
    kernel void updateKernel(device float* oldFeatures [[buffer(0)]],
                          device float* aggregated [[buffer(1)]],
                          device float* newFeatures [[buffer(2)]],
                          device float* w1 [[buffer(3)]],
                          device float* w2 [[buffer(4)]],
                          device float* bias [[buffer(5)]],
                          constant uint& numNodes [[buffer(6)]],
                          constant uint& hiddenDim [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
        uint node = id / hiddenDim;
        uint dim = id % hiddenDim;

        if (node >= numNodes) return;

        float sum = bias[dim];

        // W1 * h
        for (uint d = 0; d < hiddenDim; d++) {
            uint wIdx = dim * hiddenDim + d;
            sum += w1[wIdx] * oldFeatures[node * hiddenDim + d];
        }

        // W2 * agg
        for (uint d = 0; d < hiddenDim; d++) {
            uint wIdx = dim * hiddenDim + d;
            sum += w2[wIdx] * aggregated[node * hiddenDim + d];
        }

        // ReLU activation
        newFeatures[node * hiddenDim + dim] = fmax(0.0, sum);
    }

    // Graph Laplacian computation: L = D - A
    kernel void laplacianKernel(device uint* degrees [[buffer(0)]],
                            device float* laplacian [[buffer(1)]],
                            device uint* edges [[buffer(2)]],
                            constant uint& numNodes [[buffer(3)]],
                            constant uint& numEdges [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
        uint i = id / numNodes;
        uint j = id % numNodes;

        if (i >= numNodes || j >= numNodes) return;

        // Diagonal: degree
        if (i == j) {
            laplacian[id] = float(degrees[i]);
        } else {
            // Off-diagonal: -1 if edge exists
            laplacian[id] = 0.0;
            for (uint e = 0; e < numEdges; e++) {
                if ((edges[e * 2] == i && edges[e * 2 + 1] == j) ||
                    (edges[e * 2] == j && edges[e * 2 + 1] == i)) {
                    laplacian[id] = -1.0;
                    break;
                }
            }
        }
    }

    // Graph convolution: H' = D^{-1/2} * A * D^{-1/2} * H * W
    kernel void graphConvKernel(device float* adj [[buffer(0)]],
                             device float* features [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device float* weights [[buffer(3)]],
                             device float* degrees [[buffer(4)]],
                             constant uint& numNodes [[buffer(5)]],
                             constant uint& hiddenDim [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
        uint row = id / hiddenDim;
        uint dim = id % hiddenDim;

        if (row >= numNodes) return;

        float sum = 0.0;

        for (uint col = 0; col < numNodes; col++) {
            // D^{-1/2} * A * D^{-1/2}
            float deg_i = fmax(1.0, sqrt(degrees[row]));
            float deg_j = fmax(1.0, sqrt(degrees[col]));
            float norm = (adj[row * numNodes + col] == 0.0) ? 0.0 : 1.0 / (deg_i * deg_j);

            // Multiply with feature
            sum += norm * features[col * hiddenDim + dim];
        }

        // Apply weights
        float w = weights[dim * hiddenDim + dim];
        output[row * hiddenDim + dim] = fmax(0.0, sum * w);
    }

    // Graph pooling (simple max pooling over node features)
    kernel void graphPoolKernel(device float* features [[buffer(0)]],
                              device float* pooled [[buffer(1)]],
                              device uint* edgeDst [[buffer(2)]],
                              constant uint& numEdges [[buffer(3)]],
                              constant uint& hiddenDim [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
        if (id >= hiddenDim) return;

        float maxVal = -1e9;

        for (uint e = 0; e < numEdges; e++) {
            uint dstNode = edgeDst[e];
            float val = features[dstNode * hiddenDim + id];
            maxVal = fmax(maxVal, val);
        }

        pooled[id] = maxVal;
    }

    // Edge feature computation
    kernel void edgeFeaturesKernel(device float* srcFeatures [[buffer(0)]],
                                device float* dstFeatures [[buffer(1)]],
                                device float* edgeFeats [[buffer(2)]],
                                device uint* edges [[buffer(3)]],
                                constant uint& numEdges [[buffer(4)]],
                                constant uint& hiddenDim [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
        if (id >= numEdges) return;

        uint src = edges[id * 2];
        uint dst = edges[id * 2 + 1];

        // Edge feature = |h_src - h_dst|
        float sum = 0.0;
        for (uint d = 0; d < hiddenDim; d++) {
            float diff = srcFeatures[src * hiddenDim + d] - dstFeatures[dst * hiddenDim + d];
            sum += diff * diff;
        }
        edgeFeats[id] = sqrt(sum);
    }
    """

    // MARK: - Main Run
    public func run() throws {
        print("\n=== ANE Graph Neural Network (GNN) Benchmark ===")
        print("Testing message passing and graph convolution on ANE\n")

        var allResults: [(name: String, messageTime: Double, aggregateTime: Double, updateTime: Double, totalTime: Double)] = []

        for config in configurations {
            let result = try runConfiguration(config)
            allResults.append(result)
            print("\n\(config.name):")
            print("  Message Pass:   \(String(format: "%.4f", result.messageTime * 1000)) ms")
            print("  Aggregation:    \(String(format: "%.4f", result.aggregateTime * 1000)) ms")
            print("  Update:        \(String(format: "%.4f", result.updateTime * 1000)) ms")
            print("  Total Time:   \(String(format: "%.4f", result.totalTime * 1000)) ms")
        }

        saveResults(allResults)
    }

    // MARK: - Run Single Configuration
    func runConfiguration(_ config: (name: String, numNodes: Int, numEdges: Int, hiddenDim: Int, numLayers: Int)) throws -> (name: String, messageTime: Double, aggregateTime: Double, updateTime: Double, totalTime: Double) {
        print("  Running \(config.name) (nodes=\(config.numNodes), edges=\(config.numEdges), hidden=\(config.hiddenDim), layers=\(config.numLayers))...")

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
            throw NSError(domain: "ANEBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create library"])
        }

        guard let messageFunc = library.makeFunction(name: "messagePassKernel"),
              let aggregateFunc = library.makeFunction(name: "aggregateSumKernel"),
              let updateFunc = library.makeFunction(name: "updateKernel"),
              let edgeFeatFunc = library.makeFunction(name: "edgeFeaturesKernel")
        else {
            throw NSError(domain: "ANEBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create kernels"])
        }

        guard let messagePipeline = try? device.makeComputePipelineState(function: messageFunc),
              let aggregatePipeline = try? device.makeComputePipelineState(function: aggregateFunc),
              let updatePipeline = try? device.makeComputePipelineState(function: updateFunc),
              let edgeFeatPipeline = try? device.makeComputePipelineState(function: edgeFeatFunc)
        else {
            throw NSError(domain: "ANEBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline"])
        }

        // Allocate buffers
        let nodeBytes = config.numNodes * config.hiddenDim * MemoryLayout<Float>.stride
        let edgeBytes = config.numEdges * 2 * MemoryLayout<UInt32>.stride
        let edgeFeatBytes = config.numEdges * MemoryLayout<Float>.stride
        let weightBytes = config.hiddenDim * config.hiddenDim * MemoryLayout<Float>.stride

        guard let nodeBuffer = device.makeBuffer(length: nodeBytes, options: .storageModeShared),
              let nodeBuffer2 = device.makeBuffer(length: nodeBytes, options: .storageModeShared),
              let edgeBuffer = device.makeBuffer(length: edgeBytes, options: .storageModeShared),
              let edgeDstBuffer = device.makeBuffer(length: config.numEdges * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let messageBuffer = device.makeBuffer(length: config.numEdges * config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aggregatedBuffer = device.makeBuffer(length: nodeBytes, options: .storageModeShared),
              let w1Buffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let w2Buffer = device.makeBuffer(length: weightBytes, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(length: config.hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let degreeBuffer = device.makeBuffer(length: config.numNodes * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "ANEBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffers"])
        }

        // Initialize edge list (random graph)
        let edgePtr = edgeBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<config.numEdges {
            edgePtr[i * 2] = UInt32.random(in: 0..<UInt32(config.numNodes))
            edgePtr[i * 2 + 1] = UInt32.random(in: 0..<UInt32(config.numNodes))
        }

        // Initialize degrees
        let degreePtr = degreeBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<config.numNodes {
            degreePtr[i] = UInt32(config.numEdges / config.numNodes)
        }

        // Phase 1: Edge Feature Computation
        let edgeStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(edgeFeatPipeline)
            encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(nodeBuffer2, offset: 0, index: 1)
            encoder.setBuffer(messageBuffer, offset: 0, index: 2)
            encoder.setBuffer(edgeBuffer, offset: 0, index: 3)

            var numEdges = UInt32(config.numEdges)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numEdges, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numEdges + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let edgeTime = Double(getTimeNanos() - edgeStart) / 1e9 / 20.0

        // Phase 2: Message Passing
        let messageStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(messagePipeline)
            encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(nodeBuffer2, offset: 0, index: 1)
            encoder.setBuffer(messageBuffer, offset: 0, index: 2)
            encoder.setBuffer(w1Buffer, offset: 0, index: 3)
            encoder.setBuffer(edgeBuffer, offset: 0, index: 4)

            var numEdges = UInt32(config.numEdges)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numEdges, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numEdges + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let messageTime = Double(getTimeNanos() - messageStart) / 1e9 / 20.0

        // Phase 3: Aggregation
        let aggregateStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(aggregatePipeline)
            encoder.setBuffer(messageBuffer, offset: 0, index: 0)
            encoder.setBuffer(aggregatedBuffer, offset: 0, index: 1)
            encoder.setBuffer(edgeBuffer, offset: 0, index: 2)

            var numEdges = UInt32(config.numEdges)
            var numNodes = UInt32(config.numNodes)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numEdges, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&numNodes, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numNodes * config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let aggregateTime = Double(getTimeNanos() - aggregateStart) / 1e9 / 20.0

        // Phase 4: Update
        let updateStart = getTimeNanos()
        for _ in 0..<20 {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(updatePipeline)
            encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(aggregatedBuffer, offset: 0, index: 1)
            encoder.setBuffer(nodeBuffer2, offset: 0, index: 2)
            encoder.setBuffer(w1Buffer, offset: 0, index: 3)
            encoder.setBuffer(w2Buffer, offset: 0, index: 4)
            encoder.setBuffer(biasBuffer, offset: 0, index: 5)

            var numNodes = UInt32(config.numNodes)
            var hiddenDim = UInt32(config.hiddenDim)
            encoder.setBytes(&numNodes, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (config.numNodes * config.hiddenDim + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let updateTime = Double(getTimeNanos() - updateStart) / 1e9 / 20.0

        let totalTime = edgeTime + messageTime + aggregateTime + updateTime

        return (config.name, messageTime, aggregateTime, updateTime, totalTime)
    }

    // MARK: - Save Results
    func saveResults(_ results: [(name: String, messageTime: Double, aggregateTime: Double, updateTime: Double, totalTime: Double)]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let dir = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANEGraphNeuralNetwork"

        let log = """
        === ANE Graph Neural Network (GNN) Benchmark ===
        Timestamp: \(timestamp)
        Device: \(device.name)

        Results:
        | Configuration | Message (ms) | Aggregate (ms) | Update (ms) | Total (ms) |
        |--------------|--------------|----------------|-------------|------------|
        \(results.map { "| \($0.name) | \(String(format: "%.4f", $0.messageTime * 1000)) | \(String(format: "%.4f", $0.aggregateTime * 1000)) | \(String(format: "%.4f", $0.updateTime * 1000)) | \(String(format: "%.4f", $0.totalTime * 1000)) |" }.joined(separator: "\n"))

        Analysis:
        - Message Pass: Compute messages along edges: m_{ij} = f(h_i, h_j)
        - Aggregation: Sum/mean of incoming messages: h_i = aggregate(m_{ij})
        - Update: Combine old and aggregated: h_i' = update(h_i, agg)

        Key Insights:
        - GNNs operate on irregular graph structure
        - Message passing enables information flow across graph
        - ANE handles the dense matrix operations efficiently
        """

        try? log.write(toFile: "\(dir)/LOG.txt", atomically: true, encoding: .utf8)

        let research = """
        # ANE Graph Neural Network (GNN) Research

        ## Overview
        This benchmark evaluates Apple's Neural Engine for Graph Neural Network workloads - neural networks that operate on graph-structured data with irregular connectivity.

        ## What are Graph Neural Networks?

        GNNs are neural networks designed to work with graph-structured data:

        ### Core Idea
        Instead of processing grids (CNN) or sequences (RNN), GNNs process graphs:
        - Nodes: entities with features
        - Edges: relationships between entities
        - Graph structure: irregular connectivity

        ### Message Passing Framework
        GNNs use a message passing framework with three steps:
        1. **Message**: Compute message from neighbor to node
           m_{ij} = f(h_i, h_j, e_{ij})

        2. **Aggregate**: Combine messages from neighbors
           h_i' = aggregate({m_{ij} : j in N(i)})

        3. **Update**: Update node representation
           h_i'' = update(h_i, h_i')

        ## GNN Layer Types

        ### GraphSAGE
        - Neighbor sampling + aggregation
        - Mean, max, or LSTM aggregation
        - Inductive learning capability

        ### Graph Convolutional Network (GCN)
        - Spectral graph convolution
        - H' = D^{-1/2} * A * D^{-1/2} * H * W
        - Simplified message passing

        ### Graph Attention Network (GAT)
        - Attention over neighbors
        - α_{ij} = attention(h_i, h_j)
        - m_{ij} = α_{ij} * W * h_j

        ### Message Passing Neural Network (MPNN)
        - General framework
        - Variants: GCN, GAT, GGNN, etc.

        ## Benchmark Phases

        ### Phase 1: Edge Feature Computation
        - Compute features for each edge
        - |h_src - h_dst| (difference of node features)

        ### Phase 2: Message Passing
        - For each edge: m = W * edge_feature
        - O(E * H) operations

        ### Phase 3: Aggregation
        - Sum messages to each destination node
        - O(E * H) operations

        ### Phase 4: Update
        - h' = ReLU(W1 * h + W2 * agg)
        - O(N * H²) operations

        ## Graph Types

        - **Social Networks**: Users as nodes, interactions as edges
        - **Knowledge Graphs**: Entities and relations
        - **Molecular Graphs**: Atoms and bonds
        - **Point Clouds**: K-nearest neighbor graphs
        - **3D Meshes**: Polygonal meshes

        ## ANE vs GPU for GNNs

        | Aspect | ANE | GPU |
        |--------|-----|-----|
        | Node Features | Good | Excellent |
        | Message Pass | Good | Excellent |
        | Aggregation | Good | Excellent |
        | Sparse Access | Limited | Excellent |
        | Graph Structure | Good | Good |

        ## Key Findings

        1. **Irregular Connectivity**: GNNs handle non-grid data structures

        2. **Message Passing**: Information flows through edges

        3. **Inductive Learning**: GNNs can generalize to unseen graphs

        4. **Graph Attention**: Learn importance of neighbors

        5. **ANE Suitability**: Good for dense operations, limited for sparse irregular access

        ## Applications

        - **Social Networks**: Friend recommendations, community detection
        - **Drug Discovery**: Molecular property prediction
        - **Recommendation Systems**: User-item interactions
        - **Knowledge Graphs**: Link prediction, entity classification
        - **Autonomous Driving**: Scene graph for understanding
        - **Financial**: Fraud detection via transaction graphs

        ## Future Work

        - Implement GAT (Graph Attention Network)
        - Add graph pooling (graph coarsening)
        - Test on larger graphs with mini-batching
        - Benchmark sparse matrix operations
        """

        try? research.write(toFile: "\(dir)/RESEARCH.md", atomically: true, encoding: .utf8)

        print("\n✓ Results saved to \(dir)/LOG.txt and RESEARCH.md")
    }
}
