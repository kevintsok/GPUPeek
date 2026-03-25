import Foundation
import Metal

// MARK: - Hash Analysis Benchmark

let hashAnalysisShaders = """
#include <metal_stdlib>
using namespace metal;

// Simple hash function - murmurhash-like
inline uint hash(uint key, uint seed) {
    uint c1 = 0xcc9e2d51u;
    uint c2 = 0x1b873593u;

    uint k = key;
    k *= c1;
    k = (k << 15) | (k >> 17);
    k *= c2;

    uint h = seed;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64u;

    return h;
}

// Hash with table size
inline uint hash_table(uint key, uint tableSize, uint seed) {
    return hash(key, seed) & (tableSize - 1);
}

// Hash lookup - single hash
kernel void hash_lookup(device const uint* keys [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       device const uint* hash_table [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       constant uint& table_size [[buffer(4)]],
                       constant uint& seed [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint idx = hash_table(key, table_size, seed);
    results[id] = hash_table[idx];
}

// Hash insertion with chaining
kernel void hash_insert(device uint* keys [[buffer(0)]],
                       device uint* values [[buffer(1)]],
                       device uint* table [[buffer(2)]],
                       device uint* next [[buffer(3)]],
                       device atomic_uint* count [[buffer(4)]],
                       constant uint& size [[buffer(5)]],
                       constant uint& table_size [[buffer(6)]],
                       constant uint& seed [[buffer(7)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint val = values[id];
    uint idx = hash_table(key, table_size, seed);

    // Insert at head of chain
    uint old_head = table[idx];
    table[idx] = id;
    next[id] = old_head;
}

// Hash probe (open addressing)
kernel void hash_probe_linear(device const uint* keys [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            device const uint* hash_table [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            constant uint& table_size [[buffer(4)]],
                            constant uint& seed [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint start_idx = hash_table(key, table_size, seed);

    // Linear probing
    uint idx = start_idx;
    for (uint i = 0; i < table_size; i++) {
        if (hash_table[idx] == key) {
            results[id] = idx;
            return;
        }
        idx = (idx + 1) & (table_size - 1);
    }
    results[id] = UINT_MAX;  // Not found
}

// Bloom filter membership test
kernel void bloom_test(device const uint* keys [[buffer(0)]],
                     device uchar* results [[buffer(1)]],
                     device const uint* bit_array [[buffer(2)]],
                     constant uint& size [[buffer(3)]],
                     constant uint& num_hashes [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    bool found = true;

    for (uint h = 0; h < num_hashes; h++) {
        uint idx = hash_table(key, 32768, h * 12345);  // Different seed per hash
        if ((bit_array[idx / 32] & (1u << (idx % 32))) == 0) {
            found = false;
            break;
        }
    }
    results[id] = found ? 1 : 0;
}

// Bloom filter insertion
kernel void bloom_insert(device const uint* keys [[buffer(0)]],
                        device uint* bit_array [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        constant uint& num_hashes [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];

    for (uint h = 0; h < num_hashes; h++) {
        uint idx = hash_table(key, 32768, h * 12345);
        atomic_fetch_or_explicit(&bit_array[idx / 32], 1u << (idx % 32),
                               memory_order_relaxed, memory_scope_device);
    }
}

// Hash join - match two tables
kernel void hash_join(device const uint* keys_a [[buffer(0)]],
                     device const uint* values_a [[buffer(1)]],
                     device const uint* keys_b [[buffer(2)]],
                     device uint* results [[buffer(3)]],
                     device uint* result_count [[buffer(4)]],
                     constant uint& size_a [[buffer(5)]],
                     constant uint& size_b [[buffer(6)]],
                     constant uint& seed [[buffer(7)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= size_a) return;

    uint key = keys_a[id];
    uint val_a = values_a[id];

    // Search in B
    uint start_idx = hash_table(key, size_b, seed);
    uint idx = start_idx;
    bool found = false;

    for (uint i = 0; i < size_b; i++) {
        if (keys_b[idx] == key) {
            uint result_idx = atomic_fetch_add(result_count, 1, memory_order_relaxed);
            results[result_idx * 2] = val_a;
            results[result_idx * 2 + 1] = keys_b[idx];
            found = true;
        }
        idx = (idx + 1) & (size_b - 1);
        if (idx == start_idx) break;
    }
}
"""

public struct HashAnalysisBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Hash Analysis Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: hashAnalysisShaders, options: nil) else {
            print("Failed to compile hash analysis shaders")
            return
        }

        let sizes = [16384, 65536, 262144]

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            let tableSize = 32768  // Power of 2 for fast modulo

            guard let keysBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let valuesBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let tableBuf = device.makeBuffer(length: tableSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let resultsBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize keys and values
            let keysPtr = keysBuf.contents().bindMemory(to: UInt32.self, capacity: size)
            let valuesPtr = valuesBuf.contents().bindMemory(to: UInt32.self, capacity: size)
            for i in 0..<size {
                keysPtr[i] = UInt32(i * 17 + 12345)  // Pseudo-random keys
                valuesPtr[i] = UInt32(i * 31 + 67890)
            }

            var sizeValue = UInt32(size)
            var tableSizeValue = UInt32(tableSize)
            var seed: UInt32 = 42
            var numHashes: UInt32 = 3

            // Test hash lookup
            if let lookupFunc = library.makeFunction(name: "hash_lookup"),
               let lookupPipeline = try? device.makeComputePipelineState(function: lookupFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(lookupPipeline)
                    encoder.setBuffer(keysBuf, offset: 0, index: 0)
                    encoder.setBuffer(resultsBuf, offset: 0, index: 1)
                    encoder.setBuffer(tableBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&tableSizeValue, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.setBytes(&seed, length: MemoryLayout<UInt32>.size, index: 5)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("Hash Lookup: \(String(format: "%.2f", throughput)) GE/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test bloom filter
            let bitArraySize = 32768 / 32
            guard let bitArrayBuf = device.makeBuffer(length: bitArraySize * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let bloomResultsBuf = device.makeBuffer(length: size * MemoryLayout<UInt8>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize bit array
            let bitPtr = bitArrayBuf.contents().bindMemory(to: UInt32.self, capacity: bitArraySize)
            for i in 0..<bitArraySize {
                bitPtr[i] = 0
            }

            // Insert into bloom filter
            if let insertFunc = library.makeFunction(name: "bloom_insert"),
               let insertPipeline = try? device.makeComputePipelineState(function: insertFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    // Clear bit array
                    for i in 0..<bitArraySize {
                        bitPtr[i] = 0
                    }

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(insertPipeline)
                    encoder.setBuffer(keysBuf, offset: 0, index: 0)
                    encoder.setBuffer(bitArrayBuf, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&numHashes, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("Bloom Insert: \(String(format: "%.2f", throughput)) GE/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test bloom filter membership
            if let testFunc = library.makeFunction(name: "bloom_test"),
               let testPipeline = try? device.makeComputePipelineState(function: testFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(testPipeline)
                    encoder.setBuffer(keysBuf, offset: 0, index: 0)
                    encoder.setBuffer(bloomResultsBuf, offset: 0, index: 1)
                    encoder.setBuffer(bitArrayBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&numHashes, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("Bloom Test: \(String(format: "%.2f", throughput)) GE/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Hash tables with chaining are simple but need extra memory")
        print("2. Open addressing with linear probing has better cache behavior")
        print("3. Bloom filters provide memory-efficient set membership")
        print("4. Multiple hash functions reduce false positive rate")
        print("5. Hash join is fundamental for database operations")
    }
}
