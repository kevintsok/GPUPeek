import Foundation
import Metal

let graphShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void bfs_kernel(device const uint* edges [[buffer(0)]],
                      device const uint* adjacency [[buffer(1)]],
                      device uint* distances [[buffer(2)]],
                      device uint* frontier [[buffer(3)]],
                      device atomic_uint* visited [[buffer(4)]],
                      constant uint& numVertices [[buffer(5)]],
                      constant uint& numEdges [[buffer(6)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= numVertices) return;
    if (frontier[id] == 1) {
        uint dist = distances[id];
        for (uint i = adjacency[id]; i < adjacency[id + 1]; i++) {
            uint neighbor = edges[i];
            uint old = atomic_fetch_add_explicit(&visited[neighbor], 1, memory_order_relaxed, memory_scope_device);
            if (old == 0) {
                distances[neighbor] = dist + 1;
                frontier[neighbor] = 1;
            }
        }
        frontier[id] = 0;
    }
}
"""

public struct GraphBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Graph Algorithms Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: graphShaders, options: nil) else {
            print("Failed to compile graph shaders")
            return
        }

        let vertices = 65536
        let edges = 262144

        guard let edgesBuf = device.makeBuffer(length: edges * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let adjBuf = device.makeBuffer(length: (vertices + 1) * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let distBuf = device.makeBuffer(length: vertices * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let frontierBuf = device.makeBuffer(length: vertices * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let visitedBuf = device.makeBuffer(length: vertices * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return
        }

        var numVert = UInt32(vertices)
        var numEdg = UInt32(edges)

        print("\n--- BFS Graph Traversal ---")
        print("Vertices: \(vertices), Edges: \(edges)")

        if let bfsFunc = library.makeFunction(name: "bfs_kernel"),
           let bfsPipeline = try? device.makeComputePipelineState(function: bfsFunc) {
            let iterations = 10
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(bfsPipeline)
                encoder.setBuffer(edgesBuf, offset: 0, index: 0)
                encoder.setBuffer(adjBuf, offset: 0, index: 1)
                encoder.setBuffer(distBuf, offset: 0, index: 2)
                encoder.setBuffer(frontierBuf, offset: 0, index: 3)
                encoder.setBuffer(visitedBuf, offset: 0, index: 4)
                encoder.setBytes(&numVert, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.setBytes(&numEdg, length: MemoryLayout<UInt32>.size, index: 6)
                encoder.dispatchThreads(MTLSize(width: vertices, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(vertices) / elapsed / 1e9
            print("BFS: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. BFS performance limited by random memory access")
        print("2. Graph algorithms have irregular memory patterns")
        print("3. Frontier-based approach helps manage parallelism")
    }
}
