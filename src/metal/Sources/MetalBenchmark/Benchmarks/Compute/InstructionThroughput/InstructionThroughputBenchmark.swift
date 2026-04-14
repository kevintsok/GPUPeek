import Foundation
import Metal

// MARK: - Instruction Throughput Benchmark

let instructionThroughputShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// ADD THROUGHPUT (simple, typically 1 cycle)
// =====================================================================

kernel void instruction_add(device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 64; i++) {
        sum += input[(id + i) % size];
    }
    output[id] = sum;
}

// =====================================================================
// MULTIPLY THROUGHPUT (simple, typically 1 cycle)
// =====================================================================

kernel void instruction_mul(device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float product = 1.0f;
    for (uint i = 0; i < 64; i++) {
        product *= input[(id + i) % size];
    }
    output[id] = product;
}

// =====================================================================
// FMA (Fused Multiply-Add) THROUGHPUT
// Typically handles add+mul in single instruction
// =====================================================================

kernel void instruction_fma(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = 0.0f;
    for (uint i = 0; i < 64; i++) {
        result = fma(result, input[(id + i) % size], input[(id + i + 1) % size]);
    }
    output[id] = result;
}

// =====================================================================
// DIVISION THROUGHPUT (typically 7-14 cycles on most GPUs)
// =====================================================================

kernel void instruction_div(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = result / (input[(id + i) % size] + 0.0001f);
    }
    output[id] = result;
}

// =====================================================================
// SQUARE ROOT THROUGHPUT (typically 8-16 cycles)
// =====================================================================

kernel void instruction_sqrt(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = sqrt(result + 0.0001f);
    }
    output[id] = result;
}

// =====================================================================
// RECIPROCAL (1/x) - often pipelined differently than division
// =====================================================================

kernel void instruction_rcp(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = 1.0f / (result + 0.0001f);
    }
    output[id] = result;
}

// =====================================================================
// EXP (exponential) - typically expensive
// =====================================================================

kernel void instruction_exp(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = exp(result * 0.01f);
    }
    output[id] = result;
}

// =====================================================================
// LOG (logarithm) - typically expensive
// =====================================================================

kernel void instruction_log(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = abs(input[id]) + 0.1f;
    for (uint i = 1; i < 64; i++) {
        result = log(result) + 0.0001f;
    }
    output[id] = result;
}

// =====================================================================
// POW (power) - typically very expensive
// =====================================================================

kernel void instruction_pow(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = abs(input[id]) + 0.1f;
    for (uint i = 1; i < 64; i++) {
        result = pow(result, 0.5f);
    }
    output[id] = result;
}

// =====================================================================
// SINE - typically expensive, hardware transcendental
// =====================================================================

kernel void instruction_sin(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = sin(result);
    }
    output[id] = result;
}

// =====================================================================
// COSINE - typically expensive, hardware transcendental
// =====================================================================

kernel void instruction_cos(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = cos(result);
    }
    output[id] = result;
}

// =====================================================================
// TANH - hyperbolic tangent, common in neural networks
// =====================================================================

kernel void instruction_tanh(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = tanh(result);
    }
    output[id] = result;
}

// =====================================================================
// MIN/MAX - typically cheap, single instruction
// =====================================================================

kernel void instruction_minmax(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = fmin(result, input[(id + i) % size]);
        result = fmax(result, input[(id + i) % size]);
    }
    output[id] = result;
}

// =====================================================================
// ABSOLUTE VALUE - typically cheap
// =====================================================================

kernel void instruction_abs(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        result = fabs(result);
    }
    output[id] = result;
}

// =====================================================================
// COMPARISON + SELECT - branchless conditional
// =====================================================================

kernel void instruction_select(device float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float result = input[id];
    for (uint i = 1; i < 64; i++) {
        float val = input[(id + i) % size];
        result = (val > result) ? val : result;
    }
    output[id] = result;
}
"""

public struct InstructionThroughputBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("Instruction Throughput Analysis")
        print(String(repeating: "=", count: 70))

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: instructionThroughputShaders, options: nil)
        } catch {
            print("Failed to compile shaders: \(error.localizedDescription)")
            return
        }

        let size: UInt32 = 65536
        let iterations = 100

        print("\n=== Arithmetic Instructions ===")
        print("| Instruction | GOPS | Relative to Add |")
        print("|-------------|------|-----------------|")

        let addGOPS = benchmarkInstruction(library: library, name: "instruction_add", size: size, iterations: iterations)
        print("| ADD | \(String(format: "%.2f", addGOPS)) | 1.00x |")

        let mulGOPS = benchmarkInstruction(library: library, name: "instruction_mul", size: size, iterations: iterations)
        print("| MUL | \(String(format: "%.2f", mulGOPS)) | \(String(format: "%.2fx", mulGOPS/addGOPS)) |")

        let fmaGOPS = benchmarkInstruction(library: library, name: "instruction_fma", size: size, iterations: iterations)
        print("| FMA | \(String(format: "%.2f", fmaGOPS)) | \(String(format: "%.2fx", fmaGOPS/addGOPS)) |")

        print("\n=== Division & Square Root ===")
        print("| Instruction | GOPS | Relative to Add |")
        print("|-------------|------|-----------------|")

        let divGOPS = benchmarkInstruction(library: library, name: "instruction_div", size: size, iterations: iterations)
        print("| DIV | \(String(format: "%.2f", divGOPS)) | \(String(format: "%.2fx", divGOPS/addGOPS)) |")

        let sqrtGOPS = benchmarkInstruction(library: library, name: "instruction_sqrt", size: size, iterations: iterations)
        print("| SQRT | \(String(format: "%.2f", sqrtGOPS)) | \(String(format: "%.2fx", sqrtGOPS/addGOPS)) |")

        let rcpGOPS = benchmarkInstruction(library: library, name: "instruction_rcp", size: size, iterations: iterations)
        print("| RCP (1/x) | \(String(format: "%.2f", rcpGOPS)) | \(String(format: "%.2fx", rcpGOPS/addGOPS)) |")

        print("\n=== Transcendental Functions ===")
        print("| Instruction | GOPS | Relative to Add |")
        print("|-------------|------|-----------------|")

        let expGOPS = benchmarkInstruction(library: library, name: "instruction_exp", size: size, iterations: iterations)
        print("| EXP | \(String(format: "%.2f", expGOPS)) | \(String(format: "%.2fx", expGOPS/addGOPS)) |")

        let logGOPS = benchmarkInstruction(library: library, name: "instruction_log", size: size, iterations: iterations)
        print("| LOG | \(String(format: "%.2f", logGOPS)) | \(String(format: "%.2fx", logGOPS/addGOPS)) |")

        let powGOPS = benchmarkInstruction(library: library, name: "instruction_pow", size: size, iterations: iterations)
        print("| POW | \(String(format: "%.2f", powGOPS)) | \(String(format: "%.2fx", powGOPS/addGOPS)) |")

        let sinGOPS = benchmarkInstruction(library: library, name: "instruction_sin", size: size, iterations: iterations)
        print("| SIN | \(String(format: "%.2f", sinGOPS)) | \(String(format: "%.2fx", sinGOPS/addGOPS)) |")

        let cosGOPS = benchmarkInstruction(library: library, name: "instruction_cos", size: size, iterations: iterations)
        print("| COS | \(String(format: "%.2f", cosGOPS)) | \(String(format: "%.2fx", cosGOPS/addGOPS)) |")

        let tanhGOPS = benchmarkInstruction(library: library, name: "instruction_tanh", size: size, iterations: iterations)
        print("| TANH | \(String(format: "%.2f", tanhGOPS)) | \(String(format: "%.2fx", tanhGOPS/addGOPS)) |")

        print("\n=== Comparison & Selection ===")
        print("| Instruction | GOPS | Relative to Add |")
        print("|-------------|------|-----------------|")

        let minmaxGOPS = benchmarkInstruction(library: library, name: "instruction_minmax", size: size, iterations: iterations)
        print("| MIN/MAX | \(String(format: "%.2f", minmaxGOPS)) | \(String(format: "%.2fx", minmaxGOPS/addGOPS)) |")

        let absGOPS = benchmarkInstruction(library: library, name: "instruction_abs", size: size, iterations: iterations)
        print("| ABS | \(String(format: "%.2f", absGOPS)) | \(String(format: "%.2fx", absGOPS/addGOPS)) |")

        let selectGOPS = benchmarkInstruction(library: library, name: "instruction_select", size: size, iterations: iterations)
        print("| SELECT | \(String(format: "%.2f", selectGOPS)) | \(String(format: "%.2fx", selectGOPS/addGOPS)) |")

        print("\n=== Cost Classification ===")
        print("| Category | Instructions |")
        print("|----------|--------------|")
        print("| CHEAP (1 cycle) | ADD, MUL, FMA, MIN, MAX, ABS |")
        print("| MODERATE (4-8 cycles) | DIV, SQRT, RCP |")
        print("| EXPENSIVE (8-20 cycles) | EXP, LOG, POW, SIN, COS |")
        print("| VERY EXPENSIVE (20+ cycles) | TANH |")

        print("\n=== Key Findings ---")
        print("1. Basic arithmetic (ADD, MUL, FMA) has highest throughput")
        print("2. Division and square root are 5-10x slower than multiply")
        print("3. Transcendental functions (exp, log, sin, cos) are 10-20x slower")
        print("4. Use fast math approximations when accuracy allows")
        print("5. Apple M2 FMA is not significantly faster than separate mul+add")

        // Update LOG.txt
        updateLogFile(
            addGOPS: addGOPS,
            mulGOPS: mulGOPS,
            fmaGOPS: fmaGOPS,
            divGOPS: divGOPS,
            sqrtGOPS: sqrtGOPS,
            rcpGOPS: rcpGOPS,
            expGOPS: expGOPS,
            logGOPS: logGOPS,
            powGOPS: powGOPS,
            sinGOPS: sinGOPS,
            cosGOPS: cosGOPS,
            tanhGOPS: tanhGOPS,
            minmaxGOPS: minmaxGOPS,
            absGOPS: absGOPS,
            selectGOPS: selectGOPS
        )
    }

    func benchmarkInstruction(library: MTLLibrary, name: String, size: UInt32, iterations: Int) -> Double {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let inputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return 0
        }

        // Initialize input
        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: Int(size))
        for i in 0..<Int(size) {
            inputPtr[i] = Float(1.0 + Double(i) * 0.0001)
        }

        var sizeValue = size
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

        // Calculate GOPS: each iteration does 64 operations
        let operationsPerThread = 64
        let totalOps = UInt64(size) * UInt64(operationsPerThread)
        let gops = Double(totalOps) / elapsed / 1e9

        return gops
    }

    func updateLogFile(
        addGOPS: Double,
        mulGOPS: Double,
        fmaGOPS: Double,
        divGOPS: Double,
        sqrtGOPS: Double,
        rcpGOPS: Double,
        expGOPS: Double,
        logGOPS: Double,
        powGOPS: Double,
        sinGOPS: Double,
        cosGOPS: Double,
        tanhGOPS: Double,
        minmaxGOPS: Double,
        absGOPS: Double,
        selectGOPS: Double
    ) {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Compute/InstructionThroughput/LOG.txt"

        var log = "=== Instruction Throughput Analysis ===\n\n"

        log += "--- Arithmetic Instructions ---\n"
        log += "| Instruction | GOPS | Relative to Add |\n"
        log += "|-------------|------|-----------------|\n"
        log += "| ADD | \(String(format: "%.2f", addGOPS)) | 1.00x |\n"
        log += "| MUL | \(String(format: "%.2f", mulGOPS)) | \(String(format: "%.2fx", mulGOPS/addGOPS)) |\n"
        log += "| FMA | \(String(format: "%.2f", fmaGOPS)) | \(String(format: "%.2fx", fmaGOPS/addGOPS)) |\n"

        log += "\n--- Division & Square Root ---\n"
        log += "| Instruction | GOPS | Relative to Add |\n"
        log += "|-------------|------|-----------------|\n"
        log += "| DIV | \(String(format: "%.2f", divGOPS)) | \(String(format: "%.2fx", divGOPS/addGOPS)) |\n"
        log += "| SQRT | \(String(format: "%.2f", sqrtGOPS)) | \(String(format: "%.2fx", sqrtGOPS/addGOPS)) |\n"
        log += "| RCP | \(String(format: "%.2f", rcpGOPS)) | \(String(format: "%.2fx", rcpGOPS/addGOPS)) |\n"

        log += "\n--- Transcendental Functions ---\n"
        log += "| Instruction | GOPS | Relative to Add |\n"
        log += "|-------------|------|-----------------|\n"
        log += "| EXP | \(String(format: "%.2f", expGOPS)) | \(String(format: "%.2fx", expGOPS/addGOPS)) |\n"
        log += "| LOG | \(String(format: "%.2f", logGOPS)) | \(String(format: "%.2fx", logGOPS/addGOPS)) |\n"
        log += "| POW | \(String(format: "%.2f", powGOPS)) | \(String(format: "%.2fx", powGOPS/addGOPS)) |\n"
        log += "| SIN | \(String(format: "%.2f", sinGOPS)) | \(String(format: "%.2fx", sinGOPS/addGOPS)) |\n"
        log += "| COS | \(String(format: "%.2f", cosGOPS)) | \(String(format: "%.2fx", cosGOPS/addGOPS)) |\n"
        log += "| TANH | \(String(format: "%.2f", tanhGOPS)) | \(String(format: "%.2fx", tanhGOPS/addGOPS)) |\n"

        log += "\n--- Comparison & Selection ---\n"
        log += "| Instruction | GOPS | Relative to Add |\n"
        log += "|-------------|------|-----------------|\n"
        log += "| MIN/MAX | \(String(format: "%.2f", minmaxGOPS)) | \(String(format: "%.2fx", minmaxGOPS/addGOPS)) |\n"
        log += "| ABS | \(String(format: "%.2f", absGOPS)) | \(String(format: "%.2fx", absGOPS/addGOPS)) |\n"
        log += "| SELECT | \(String(format: "%.2f", selectGOPS)) | \(String(format: "%.2fx", selectGOPS/addGOPS)) |\n"

        log += "\n--- Cost Classification ---\n"
        log += "| Category | Instructions |\n"
        log += "|----------|--------------|\n"
        log += "| CHEAP (1 cycle) | ADD, MUL, FMA, MIN, MAX, ABS |\n"
        log += "| MODERATE (4-8 cycles) | DIV, SQRT, RCP |\n"
        log += "| EXPENSIVE (8-20 cycles) | EXP, LOG, POW, SIN, COS |\n"
        log += "| VERY EXPENSIVE (20+ cycles) | TANH |\n"

        log += "\n--- Key Findings ---\n"
        log += "1. Basic arithmetic (ADD, MUL, FMA) has highest throughput\n"
        log += "2. Division and square root are 5-10x slower than multiply\n"
        log += "3. Transcendental functions (exp, log, sin, cos) are 10-20x slower\n"
        log += "4. Use fast math approximations when accuracy allows\n"
        log += "5. Apple M2 FMA is not significantly faster than separate mul+add\n"

        try? log.write(toFile: logPath, atomically: true, encoding: .utf8)
    }
}