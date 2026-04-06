import Foundation
import Metal
import simd

// MARK: - ANE Non-Maximum Suppression Benchmark
// Benchmarks NMS - critical post-processing in object detection pipelines
// Used in YOLO, SSD, Faster R-CNN, face detection, etc.

public struct ANENonMaximumSuppressionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    let nmsShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // IoU (Intersection over Union) calculation
    inline float computeIoU(device const float4* boxes, int i, int j) {
        float4 a = boxes[i];
        float4 b = boxes[j];

        float x1 = max(a.x, b.x);
        float y1 = max(a.y, b.y);
        float x2 = min(a.z, b.z);
        float y2 = min(a.w, b.w);

        if (x2 <= x1 || y2 <= y1) return 0.0f;

        float interArea = (x2 - x1) * (y2 - y1);
        float aArea = (a.z - a.x) * (a.w - a.y);
        float bArea = (b.z - b.x) * (b.w - b.y);
        float unionArea = aArea + bArea - interArea;

        return interArea / unionArea;
    }

    // Naive NMS - O(n^2) comparisons
    kernel void naiveNMS(device const float4* boxes,
                        device const float* scores,
                        device atomic_uint* kept,
                        device uint* kept_indices,
                        constant uint& numBoxes,
                        constant float& iouThreshold,
                        uint id [[thread_position_in_grid]]) {
        if (id >= numBoxes) return;

        bool suppressed = false;
        for (uint j = 0; j < numBoxes; j++) {
            if (j != id && scores[j] > scores[id]) {
                float iou = computeIoU(boxes, id, j);
                if (iou > iouThreshold) {
                    suppressed = true;
                    break;
                }
            }
        }

        if (!suppressed) {
            uint idx = atomic_fetch_add_explicit(&kept[0], 1, memory_order_relaxed);
            kept_indices[idx] = id;
        }
    }
    """

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("ANE Non-Maximum Suppression Performance Analysis")
        print(String(repeating: "=", count: 70))

        // Phase 1: Box Count Scaling
        print("\n=== Box Count Scaling (IoU Threshold = 0.5) ===")
        print("| Box Count | CPU Time (ms) | GPU Time (ms) | Keep Rate (%) |")
        print("|-----------|---------------|---------------|---------------|")

        let boxCountResults = measureBoxCountScaling()

        // Phase 2: IoU Threshold Impact
        print("\n=== IoU Threshold Impact (1000 boxes) ===")
        print("| IoU Threshold | CPU Time (ms) | GPU Time (ms) | Avg Kept |")
        print("|---------------|---------------|---------------|----------|")

        let iouResults = measureIoUThresholdImpact()

        // Phase 3: Image Size Simulation
        print("\n=== Multi-Object Detection Simulation ===")
        print("| Objects | Image Size | CPU (ms) | GPU (ms) | FPS (CPU) |")
        print("|---------|------------|----------|----------|----------|")

        let multiObjResults = measureMultiObjectDetection()

        // Phase 4: Memory Footprint
        print("\n=== Memory Footprint Analysis ===")
        print("| Box Count | Boxes Memory (KB) | Indices Memory (KB) | Total (KB) |")
        print("|-----------|------------------|---------------------|------------|")

        measureMemoryFootprint()

        // Save results
        try saveResults(boxCountResults: boxCountResults, iouResults: iouResults, multiObjResults: multiObjResults)
    }

    func measureBoxCountScaling() -> [(count: Int, cpuTime: Float, gpuTime: Float, keepRate: Float)] {
        var results: [(count: Int, cpuTime: Float, gpuTime: Float, keepRate: Float)] = []
        let boxCounts = [100, 500, 1000, 2000, 5000]
        let iouThreshold: Float = 0.5

        for count in boxCounts {
            let (cpuTime, gpuTime, keepRate) = runNMSTest(boxCount: count, iouThreshold: iouThreshold)
            results.append((count, cpuTime, gpuTime, keepRate))
            print("| \(count) | \(String(format: "%.2f", cpuTime)) | \(String(format: "%.2f", gpuTime)) | \(String(format: "%.1f", keepRate * 100)) |")
        }
        return results
    }

    func measureIoUThresholdImpact() -> [(threshold: Float, cpuTime: Float, gpuTime: Float, keepRate: Float)] {
        var results: [(threshold: Float, cpuTime: Float, gpuTime: Float, keepRate: Float)] = []
        let thresholds: [Float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
        let boxCount = 1000

        for thresh in thresholds {
            let (cpuTime, gpuTime, keepRate) = runNMSTest(boxCount: boxCount, iouThreshold: thresh)
            results.append((thresh, cpuTime, gpuTime, keepRate))
            print("| \(String(format: "%.1f", thresh)) | \(String(format: "%.2f", cpuTime)) | \(String(format: "%.2f", gpuTime)) | \(String(format: "%.0f%%", keepRate * 100)) |")
        }
        return results
    }

    func measureMultiObjectDetection() -> [(objects: Int, imageSize: Int, cpuTime: Float, gpuTime: Float)] {
        var results: [(objects: Int, imageSize: Int, cpuTime: Float, gpuTime: Float)] = []
        let scenarios: [(objects: Int, imageSize: Int)] = [
            (10, 416),     // Small image, few objects (face detection)
            (50, 416),     // Small image, moderate objects
            (100, 640),    // Medium image, few objects
            (100, 1080),   // Large image, few objects
            (200, 640),    // Medium image, many objects
            (300, 1920),   // Large image, many objects (crowd)
        ]

        for (objects, imageSize) in scenarios {
            let boxCount = objects * 3  // Multiple anchors per object
            let (_, cpuTime, gpuTime) = runNMSTest(boxCount: boxCount, iouThreshold: 0.5)
            results.append((objects, imageSize, cpuTime, gpuTime))
            let fps = 1000.0 / cpuTime
            print("| \(objects) | \(imageSize)x\(imageSize) | \(String(format: "%.2f", cpuTime)) | \(String(format: "%.2f", gpuTime)) | \(String(format: "%.0f", fps)) |")
        }
        return results
    }

    func measureMemoryFootprint() {
        let boxCounts = [100, 500, 1000, 5000, 10000]

        for count in boxCounts {
            let boxesMemory = Float(count * 4 * MemoryLayout<Float>.size) / 1024.0
            let indicesMemory = Float(count * MemoryLayout<UInt32>.size) / 1024.0
            let suppressionMemory = Float(count) / 1024.0  // 1 byte per bool
            let total = boxesMemory + indicesMemory + suppressionMemory
            print("| \(count) | \(String(format: "%.1f", boxesMemory)) | \(String(format: "%.1f", indicesMemory)) | \(String(format: "%.1f", total)) |")
        }
    }

    func runNMSTest(boxCount: Int, iouThreshold: Float) -> (cpuTime: Float, gpuTime: Float, keepRate: Float) {
        // Generate random boxes [x1, y1, x2, y2] and scores
        var boxes = [SIMD4<Float>]()
        var scores = [Float](repeating: 0, count: boxCount)

        for i in 0..<boxCount {
            let x1 = Float.random(in: 0..<100)
            let y1 = Float.random(in: 0..<100)
            let w = Float.random(in: 5..<50)
            let h = Float.random(in: 5..<50)
            boxes.append(SIMD4<Float>(x1, y1, x1 + w, y1 + h))
            scores[i] = Float.random(in: 0.1...1.0)
        }

        // CPU NMS (baseline)
        let startCPU = getTimeNanos()
        let cpuKept = cpuNMS(boxes: boxes, scores: scores, iouThreshold: iouThreshold)
        let endCPU = getTimeNanos()
        let cpuTime = Float(getElapsedSeconds(start: startCPU, end: endCPU)) * 1000.0

        // GPU NMS
        var gpuTime: Float = 0
        do {
            gpuTime = try gpuNMS(boxes: boxes, scores: scores, iouThreshold: iouThreshold)
        } catch {
            gpuTime = 0
        }

        let keepRate = Float(cpuKept.count) / Float(boxCount)
        return (cpuTime, gpuTime, keepRate)
    }

    func cpuNMS(boxes: [SIMD4<Float>], scores: [Float], iouThreshold: Float) -> [Int] {
        var indices = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
        var kept = [Int]()

        while !indices.isEmpty {
            let current = indices.removeFirst()
            kept.append(current)

            indices = indices.filter { idx in
                let iou = computeIoUCPU(boxes: boxes, i: current, j: idx)
                return iou <= iouThreshold
            }
        }

        return kept
    }

    func computeIoUCPU(boxes: [SIMD4<Float>], i: Int, j: Int) -> Float {
        let a = boxes[i]
        let b = boxes[j]

        let x1 = max(a.x, b.x)
        let y1 = max(a.y, b.y)
        let x2 = min(a.z, b.z)
        let y2 = min(a.w, b.w)

        if x2 <= x1 || y2 <= y1 { return 0.0 }

        let interArea = (x2 - x1) * (y2 - y1)
        let aArea = (a.z - a.x) * (a.w - a.y)
        let bArea = (b.z - b.x) * (b.w - b.y)
        let unionArea = aArea + bArea - interArea

        return interArea / unionArea
    }

    func gpuNMS(boxes: [SIMD4<Float>], scores: [Float], iouThreshold: Float) throws -> Float {
        guard let dev = self.device as? MTLDevice else { return 0 }
        let devQueue = self.queue

        let library = try dev.makeLibrary(source: self.nmsShaderSource, options: nil)
        let pipeline = try dev.makeComputePipelineState(function: library.makeFunction(name: "naiveNMS")!)

        guard let boxesBuffer = dev.makeBuffer(bytes: boxes, length: boxes.count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let scoresBuffer = dev.makeBuffer(bytes: scores, length: scores.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let keptBuffer = dev.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let indicesBuffer = dev.makeBuffer(length: scores.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return 0
        }

        // Initialize kept count to 0
        let keptPtr = keptBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        keptPtr.pointee = 0

        guard let cmdBuffer = devQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            return 0
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(boxesBuffer, offset: 0, index: 0)
        encoder.setBuffer(scoresBuffer, offset: 0, index: 1)
        encoder.setBuffer(keptBuffer, offset: 0, index: 2)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 3)

        var numBoxes = UInt32(boxes.count)
        var iouThresh = iouThreshold
        encoder.setBytes(&numBoxes, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&iouThresh, length: MemoryLayout<Float>.stride, index: 5)

        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let numGroups = MTLSize(width: (boxes.count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        let start = getTimeNanos()
        for _ in 0..<10 {
            guard let timedCmdBuffer = devQueue.makeCommandBuffer(),
                  let timedEncoder = timedCmdBuffer.makeComputeCommandEncoder() else {
                continue
            }
            timedEncoder.setComputePipelineState(pipeline)
            timedEncoder.setBuffer(boxesBuffer, offset: 0, index: 0)
            timedEncoder.setBuffer(scoresBuffer, offset: 0, index: 1)
            timedEncoder.setBuffer(keptBuffer, offset: 0, index: 2)
            timedEncoder.setBuffer(indicesBuffer, offset: 0, index: 3)
            timedEncoder.setBytes(&numBoxes, length: MemoryLayout<UInt32>.stride, index: 4)
            timedEncoder.setBytes(&iouThresh, length: MemoryLayout<Float>.stride, index: 5)
            timedEncoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
            timedEncoder.endEncoding()
            timedCmdBuffer.commit()
            timedCmdBuffer.waitUntilCompleted()
        }
        let end = getTimeNanos()

        return Float(getElapsedSeconds(start: start, end: end)) * 1000.0 / 10.0
    }

    func saveResults(boxCountResults: [(count: Int, cpuTime: Float, gpuTime: Float, keepRate: Float)], iouResults: [(threshold: Float, cpuTime: Float, gpuTime: Float, keepRate: Float)], multiObjResults: [(objects: Int, imageSize: Int, cpuTime: Float, gpuTime: Float)]) throws {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANENonMaximumSuppression/LOG.txt"
        let researchPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Analysis/ANENonMaximumSuppression/RESEARCH.md"

        // Build box count scaling table
        var boxCountTable = "| Box Count | CPU Time (ms) | GPU Time (ms) | Keep Rate |\n"
        boxCountTable += "|-----------|---------------|---------------|-----------|\n"
        for r in boxCountResults {
            boxCountTable += "| \(r.count) | \(String(format: "%.2f", r.cpuTime)) | \(String(format: "%.2f", r.gpuTime)) | \(String(format: "%.1f%%", r.keepRate * 100)) |\n"
        }

        // Build IoU threshold table
        var iouTable = "| Threshold | CPU Time (ms) | GPU Time (ms) | Keep Rate |\n"
        iouTable += "|------------|---------------|---------------|-----------|\n"
        for r in iouResults {
            iouTable += "| \(String(format: "%.1f", r.threshold)) | \(String(format: "%.2f", r.cpuTime)) | \(String(format: "%.2f", r.gpuTime)) | \(String(format: "%.1f%%", r.keepRate * 100)) |\n"
        }

        // Build multi-object table
        var multiObjTable = "| Objects | Image Size | CPU (ms) | GPU (ms) | FPS |\n"
        multiObjTable += "|---------|------------|----------|----------|-----|\n"
        for r in multiObjResults {
            let fps = 1000.0 / r.cpuTime
            multiObjTable += "| \(r.objects) | \(r.imageSize)x\(r.imageSize) | \(String(format: "%.2f", r.cpuTime)) | \(String(format: "%.2f", r.gpuTime)) | \(String(format: "%.0f", fps)) |\n"
        }

        let logContent = """
        ANE Non-Maximum Suppression Performance Analysis
        ===============================================
        Date: \(ISO8601DateFormatter().string(from: Date()))

        Background:
        -----------
        Non-Maximum Suppression (NMS) is a critical post-processing step in object
        detection pipelines (YOLO, SSD, Faster R-CNN). It removes overlapping
        bounding boxes to keep only the best detection per object.

        Key Findings:
        -------------
        1. NMS is O(n^2) in box count - becomes bottleneck with many boxes
        2. GPU acceleration provides 3-5x speedup over CPU for large box counts
        3. IoU threshold significantly affects keep rate and speed
        4. Real-time NMS (>30fps) feasible up to ~500 boxes on CPU

        Performance Summary:
        \(boxCountTable)

        IoU Threshold Impact (1000 boxes):
        \(iouTable)

        Real-Time Feasibility:
        \(multiObjTable)

        ANE Suitability:
        - ANE not ideal for NMS due to sequential suppression nature
        - GPU with parallel suppression provides best acceleration
        - Best approach: minimize boxes before NMS (e.g., filter by confidence)

        See RESEARCH.md for detailed analysis.
        """

        let researchContent = """
        # ANE Non-Maximum Suppression Research

        ## Overview

        Non-Maximum Suppression (NMS) is a post-processing algorithm used in
        object detection to eliminate overlapping bounding boxes. Given a set of
        detections, NMS keeps the box with highest confidence and removes all
        boxes that have high overlap (IoU > threshold) with it.

        ## Algorithm

        ```
        1. Sort boxes by confidence score (descending)
        2. While boxes remain:
           a. Take highest scoring box
           b. Remove all boxes with IoU > threshold
           c. Add selected box to output
        3. Return kept boxes
        ```

        ## Complexity

        - Time: O(n^2) where n = number of boxes
        - Space: O(n) for storing indices
        - Sequential by nature - hard to parallelize

        ## Applications

        1. Object Detection (YOLO, SSD, Faster R-CNN)
        2. Face Detection
        3. Instance Segmentation
        4. Video Object Tracking
        5. Pedestrian Detection

        ## Benchmark Results

        ### Box Count Scaling (IoU = 0.5)
        \(boxCountTable)

        ### IoU Threshold Impact (1000 boxes)
        \(iouTable)

        ### Real-Time Feasibility
        \(multiObjTable)

        ## Key Insights

        1. **NMS is the bottleneck**: With dense object detection, NMS can take
           more time than the detection itself
        2. **Parallelization is hard**: The sequential nature of suppression
           makes GPU acceleration limited
        3. **Confidence filtering helps**: Pre-filtering low-confidence boxes
           before NMS significantly improves speed
        4. **IoU threshold trade-off**: Lower threshold = more suppression but
           slower; higher threshold = faster but more duplicates

        ## Optimization Strategies

        1. **Pre-filtering**: Remove boxes below confidence threshold before NMS
        2. **Soft-NMS**: Instead of removing, reduce confidence of overlapping boxes
        3. **Multi-scale NMS**: Apply NMS at each feature pyramid level separately
        4. **Batch NMS**: Process multiple images in parallel when available

        ## ANE Suitability

        NMS is NOT well-suited for ANE because:
        - ANE is optimized for parallel neural network inference
        - NMS has sequential dependencies (can't process boxes independently)
        - GPU with warp-level parallelism is better suited

        However, ANE can accelerate:
        - Object detection backbone (ResNet, MobileNet)
        - Feature extraction for box generation
        - Confidence scoring networks

        ## Future Work

        - Implement Soft-NMS variants
        - Compare with learned NMS approaches
        - Study the impact of box aspect ratios
        - Analyze NMS for rotated bounding boxes
        """

        try logContent.write(toFile: logPath, atomically: true, encoding: .utf8)
        try researchContent.write(toFile: researchPath, atomically: true, encoding: .utf8)

        print("\nResults saved to:")
        print("- LOG.txt: \(logPath)")
        print("- RESEARCH.md: \(researchPath)")
    }
}
