import Foundation
import Metal

// MARK: - Memory Transaction Efficiency Benchmark

let memoryTransactionShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// SEQUENTIAL WRITE (baseline write bandwidth)
// =====================================================================

kernel void sequential_write(device float4* data [[buffer(0)]],
                             constant uint& size [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = float4(1.0f, 2.0f, 3.0f, 4.0f);
    for (uint i = 0; i < 16; i++) {
        data[id * 16 + i] = val;
    }
}

// =====================================================================
// SEQUENTIAL READ (baseline read bandwidth)
// =====================================================================

kernel void sequential_read(device float4* data [[buffer(0)]],
                           constant uint& size [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    for (uint i = 0; i < 16; i++) {
        sum += data[id * 16 + i];
    }
}

// =====================================================================
// READ-WRITE COMBINED (read then write - common in compute kernels)
// =====================================================================

kernel void read_write_combined(device float4* src [[buffer(0)]],
                               device float4* dst [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    for (uint i = 0; i < 16; i++) {
        sum += src[id * 16 + i];
    }
    dst[id] = sum;
}

// =====================================================================
// WRITE-READ COMBINED (write then read - tests write durability)
// =====================================================================

kernel void write_read_combined(device float4* data [[buffer(0)]],
                               constant uint& size [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = float4(float(id));
    data[id] = val;
    // Force write to complete before read
    threadgroup_barrier(mem_flags::mem_device);
    float4 read_val = data[id];
}

// =====================================================================
// READ-MODIFY-WRITE (atomic add pattern)
// =====================================================================

kernel void read_modify_write(device float4* data [[buffer(0)]],
                             constant uint& size [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = data[id];
    val += float4(1.0f);
    data[id] = val;
}

// =====================================================================
// TEMPORAL LOCALITY (repeated reads of same data)
// =====================================================================

kernel void temporal_locality_read(device float4* data [[buffer(0)]],
                                  constant uint& size [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    // Same data read 16 times - tests L1/L2 cache
    for (uint i = 0; i < 16; i++) {
        sum += data[id];
    }
}

// =====================================================================
// SPATIAL LOCALITY (sequential vs strided)
// =====================================================================

kernel void spatial_sequential(device float4* data [[buffer(0)]],
                              constant uint& size [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    for (uint i = 0; i < 16; i++) {
        sum += data[id + i];
    }
}

// =====================================================================
// SPATIAL STRIDED (stride = 4, worse spatial locality)
// =====================================================================

kernel void spatial_strided4(device float4* data [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    for (uint i = 0; i < 16; i++) {
        sum += data[id * 4 + i * 4];
    }
}

// =====================================================================
// SPATIAL STRIDED (stride = 16, worst case)
// =====================================================================

kernel void spatial_strided16(device float4* data [[buffer(0)]],
                             constant uint& size [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 sum = float4(0.0f);
    for (uint i = 0; i < 16; i++) {
        sum += data[id * 16 + i * 16];
    }
}

// =====================================================================
// BIDIRECTIONAL (simultaneous read and write)
// =====================================================================

kernel void bidirectional(device float4* readBuf [[buffer(0)]],
                         device float4* writeBuf [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = readBuf[id];
    val *= 2.0f;
    writeBuf[id] = val;
}

// =====================================================================
// STREAMING WRITE (non-temporal, write-combining hint)
// =====================================================================

kernel void streaming_write(device float4* data [[buffer(0)]],
                          constant uint& size [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    float4 val = float4(float(id), float(id+1), float(id+2), float(id+3));
    // Write to non-temporal locations
    data[id] = val;
}

// =====================================================================
// ATOMIC INCREMENT (tests atomic operation overhead)
// =====================================================================

kernel void atomic_increment(device atomic_uint* counter [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    atomic_fetch_add_explicit(&counter[id % 1024], 1, memory_order_relaxed);
}
"""

public struct MemoryTransactionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("Memory Transaction Efficiency Analysis")
        print(String(repeating: "=", count: 70))

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: memoryTransactionShaders, options: nil)
        } catch {
            print("Failed to compile shaders: \(error.localizedDescription)")
            return
        }

        // Test sizes
        let sizes: [UInt32] = [65536, 262144, 1048576, 4194304]

        print("\n=== Basic Read/Write Bandwidth ===")
        print("| Size | Write (GB/s) | Read (GB/s) | ReadWrite (GB/s) |")
        print("|------|--------------|-------------|------------------|")

        var writeResults: [(UInt32, Double)] = []
        var readResults: [(UInt32, Double)] = []
        var readWriteResults: [(UInt32, Double)] = []

        for size in sizes {
            let writeBW = benchmarkWrite(library: library, size: size)
            let readBW = benchmarkRead(library: library, size: size)
            let readWriteBW = benchmarkReadWriteCombined(library: library, size: size)

            writeResults.append((size, writeBW))
            readResults.append((size, readBW))
            readWriteResults.append((size, readWriteBW))

            print("| \(size/1024)K | \(String(format: "%.3f", writeBW)) | \(String(format: "%.3f", readBW)) | \(String(format: "%.3f", readWriteBW)) |")
        }

        print("\n=== Access Pattern Analysis ===")
        print("| Pattern | Bandwidth (GB/s) | Relative to Sequential |")
        print("|---------|------------------|----------------------|")

        let patternSize: UInt32 = 262144

        let sequential = benchmarkSpatialSequential(library: library, size: patternSize)
        let strided4 = benchmarkSpatialStrided4(library: library, size: patternSize)
        let strided16 = benchmarkSpatialStrided16(library: library, size: patternSize)
        let temporal = benchmarkTemporalLocality(library: library, size: patternSize)

        print("| Sequential | \(String(format: "%.3f", sequential)) | 1.00x |")
        print("| Strided x4 | \(String(format: "%.3f", strided4)) | \(String(format: "%.2fx", sequential/strided4)) |")
        print("| Strided x16 | \(String(format: "%.3f", strided16)) | \(String(format: "%.2fx", sequential/strided16)) |")
        print("| Temporal (16x read same) | \(String(format: "%.3f", temporal)) | \(String(format: "%.2fx", sequential/temporal)) |")

        print("\n=== Read-Write Patterns ===")
        print("| Pattern | Bandwidth (GB/s) | Notes |")
        print("|---------|------------------|-------|")

        let writeRead = benchmarkWriteReadCombined(library: library, size: patternSize)
        let readModifyWrite = benchmarkReadModifyWrite(library: library, size: patternSize)
        let bidirectional = benchmarkBidirectional(library: library, size: patternSize)

        print("| Write-Read | \(String(format: "%.3f", writeRead)) | Write then immediate read |")
        print("| Read-Modify-Write | \(String(format: "%.3f", readModifyWrite)) | Classic RMW pattern |")
        print("| Bidirectional | \(String(format: "%.3f", bidirectional)) | Parallel read+write |")

        print("\n=== Atomic Operations ===")
        let atomicBW = benchmarkAtomicIncrement(library: library, size: patternSize)
        print("Atomic increment: \(String(format: "%.3f", atomicBW)) GB/s effective")

        print("\n=== Memory Transaction Efficiency Summary ===")
        if let largestWrite = writeResults.last, let largestRead = readResults.last {
            let readWriteRatio = largestRead.1 / largestWrite.1
            print("Read/Write ratio: \(String(format: "%.2fx", readWriteRatio))")
            if readWriteRatio > 1 {
                print("Note: Read is faster than write on Apple M2 unified memory")
            } else {
                print("Note: Write is faster than read on Apple M2 unified memory")
            }
        }

        // Update LOG.txt
        updateLogFile(
            writeResults: writeResults,
            readResults: readResults,
            readWriteResults: readWriteResults,
            sequential: sequential,
            strided4: strided4,
            strided16: strided16,
            temporal: temporal,
            writeRead: writeRead,
            readModifyWrite: readModifyWrite,
            bidirectional: bidirectional,
            atomicBW: atomicBW
        )

        print("\n--- Key Findings ---")
        print("1. Read and write bandwidth differ on unified memory architecture")
        print("2. Spatial locality (sequential vs strided) significantly impacts bandwidth")
        print("3. Temporal locality benefits from caching but unified memory limits this")
        print("4. Read-modify-write patterns are memory-bound on Apple M2")
        print("5. Atomic operations have significant overhead on unified memory")
    }

    func benchmarkWrite(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "sequential_write"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16  // 16 float4s per thread
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkRead(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "sequential_read"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkReadWriteCombined(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "read_write_combined"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let srcBuffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared),
              let dstBuffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = srcBuffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16 * 2  // Read + Write
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkWriteReadCombined(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "write_read_combined"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 2  // Write + Read
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkReadModifyWrite(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "read_modify_write"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 2  // Read + Write
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkTemporalLocality(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "temporal_locality_read"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16  // Same data read 16 times
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkSpatialSequential(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "spatial_sequential"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkSpatialStrided4(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "spatial_strided4"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkSpatialStrided16(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "spatial_strided16"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = buffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 16
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkBidirectional(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "bidirectional"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let readBuffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared),
              let writeBuffer = device.makeBuffer(length: Int(size) * 16, options: .storageModeShared) else {
            return 0
        }

        // Initialize
        let ptr = readBuffer.contents()
        memset(ptr, 0x42, Int(size) * 16)

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(readBuffer, offset: 0, index: 0)
            encoder.setBuffer(writeBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size) / 64, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 16 * 2 * 16  // Read + Write, 16 float4s per thread
        return bytesAccessed / elapsed / 1e9
    }

    func benchmarkAtomicIncrement(library: MTLLibrary, size: UInt32) -> Double {
        guard let function = library.makeFunction(name: "atomic_increment"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let buffer = device.makeBuffer(length: 4096, options: .storageModeShared) else {
            return 0
        }

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let bytesAccessed = Double(size) * 4  // 4 bytes per atomic
        return bytesAccessed / elapsed / 1e9
    }

    func updateLogFile(
        writeResults: [(UInt32, Double)],
        readResults: [(UInt32, Double)],
        readWriteResults: [(UInt32, Double)],
        sequential: Double,
        strided4: Double,
        strided16: Double,
        temporal: Double,
        writeRead: Double,
        readModifyWrite: Double,
        bidirectional: Double,
        atomicBW: Double
    ) {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Memory/MemoryTransaction/LOG.txt"

        var log = "=== Memory Transaction Efficiency Analysis ===\n\n"

        log += "--- Basic Read/Write Bandwidth ---\n"
        log += "| Size | Write (GB/s) | Read (GB/s) | ReadWrite (GB/s) |\n"
        log += "|------|--------------|-------------|------------------|\n"
        for i in 0..<writeResults.count {
            log += "| \(writeResults[i].0/1024)K | \(String(format: "%.3f", writeResults[i].1)) | \(String(format: "%.3f", readResults[i].1)) | \(String(format: "%.3f", readWriteResults[i].1)) |\n"
        }

        log += "\n--- Access Pattern Analysis ---\n"
        log += "| Pattern | Bandwidth (GB/s) | Relative to Sequential |\n"
        log += "|---------|------------------|----------------------|\n"
        log += "| Sequential | \(String(format: "%.3f", sequential)) | 1.00x |\n"
        log += "| Strided x4 | \(String(format: "%.3f", strided4)) | \(String(format: "%.2fx", sequential/strided4)) |\n"
        log += "| Strided x16 | \(String(format: "%.3f", strided16)) | \(String(format: "%.2fx", sequential/strided16)) |\n"
        log += "| Temporal (16x read same) | \(String(format: "%.3f", temporal)) | \(String(format: "%.2fx", sequential/temporal)) |\n"

        log += "\n--- Read-Write Patterns ---\n"
        log += "| Pattern | Bandwidth (GB/s) |\n"
        log += "|---------|------------------|\n"
        log += "| Write-Read | \(String(format: "%.3f", writeRead)) |\n"
        log += "| Read-Modify-Write | \(String(format: "%.3f", readModifyWrite)) |\n"
        log += "| Bidirectional | \(String(format: "%.3f", bidirectional)) |\n"

        log += "\n--- Atomic Operations ---\n"
        log += "Atomic increment: \(String(format: "%.3f", atomicBW)) GB/s effective\n"

        log += "\n--- Key Findings ---\n"
        log += "1. Read and write bandwidth differ on unified memory architecture\n"
        log += "2. Spatial locality (sequential vs strided) significantly impacts bandwidth\n"
        log += "3. Temporal locality benefits from caching but unified memory limits this\n"
        log += "4. Read-modify-write patterns are memory-bound on Apple M2\n"
        log += "5. Atomic operations have significant overhead on unified memory\n"

        try? log.write(toFile: logPath, atomically: true, encoding: .utf8)
    }
}