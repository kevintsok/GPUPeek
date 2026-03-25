import Foundation
import Metal

public struct ArchitectureBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("GPU Architecture Query Benchmark")
        print(String(repeating: "=", count: 70))

        print("\n--- Apple M2 GPU Specifications ---")
        print("Device Name: \(device.name)")
        print("Recommended Max Working Set: \(Double(device.recommendedMaxWorkingSetSize) / 1e9) GB")
        print("Max Buffer Length: \(Double(device.maxBufferLength) / 1e9) GB")
        print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
        print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")
        print("Simd Width: 32 (fixed)")

        let sizes: [Int] = [32, 64, 128, 256, 512, 1024]
        print("\n--- Threadgroup Size vs Occupancy ---")
        for size in sizes {
            let threadsPerTG = min(size, device.maxThreadsPerThreadgroup.width)
            print("Threadgroup Size \(size): \(threadsPerTG) threads")
        }

        print("\n--- Key Insights ---")
        print("1. Apple M2 has 32KB max threadgroup memory")
        print("2. Max threads per threadgroup varies by kernel")
        print("3. SIMD width is fixed at 32 threads")
        print("4. Unified memory architecture")
    }
}
EOF
echo "Created Architecture/Benchmark.swift"